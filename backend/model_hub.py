"""
模型中心：多源搜索、下载、许可证检查、本地模型库管理

支持的数据源：
- Hugging Face Hub
- ModelScope（魔搭社区）
"""

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import huggingface_hub
from huggingface_hub import (
    HfApi,
    ModelInfo,
    hf_hub_download,
    list_models,
    scan_cache_dir,
)
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# License Classification
# ============================================================

COMMERCIAL_LICENSES = {
    "mit", "apache-2.0", "apache 2.0", "bsd", "bsd-2-clause",
    "bsd-3-clause", "unlicense", "cc0-1.0", "wtfpl",
    "openrail", "openrail++", "bigscience-openrail-m",
    "bigscience-bloom-rail-1.0", "creativeml-openrail-m",
    "gemma", "llama2", "llama3", "llama3.1", "llama3.2",
    "qwen", "deepseek", "yi",
}

RESTRICTED_LICENSES = {
    "cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-3.0",
    "cc-by-nc-sa-3.0", "research-only", "non-commercial",
    "cc-by-nc-2.0", "cc-by-nc-sa-2.0",
}

FORBIDDEN_LICENSES = {
    "cc-by-nd-4.0", "cc-by-nd-3.0", "no-derivatives",
    "proprietary", "all-rights-reserved",
}


def classify_license(license_str: str | None) -> dict:
    """
    将许可证字符串分类为：可商用 / 需审查 / 不可商用

    Returns:
        {
            "label": "🟢 可商用" | "🟡 需审查" | "🔴 不可商用" | "⚪ 未知",
            "class": "commercial" | "restricted" | "forbidden" | "unknown",
            "raw": original_string,
        }
    """
    if not license_str:
        return {"label": "⚪ 未知", "class": "unknown", "raw": license_str}

    normalized = license_str.lower().strip()

    # Check forbidden first
    for forbidden in FORBIDDEN_LICENSES:
        if forbidden in normalized:
            return {"label": "🔴 不可商用", "class": "forbidden", "raw": license_str}

    # Check restricted
    for restricted in RESTRICTED_LICENSES:
        if restricted in normalized:
            return {"label": "🟡 需审查", "class": "restricted", "raw": license_str}

    # Check commercial
    for commercial in COMMERCIAL_LICENSES:
        if commercial in normalized:
            return {"label": "🟢 可商用", "class": "commercial", "raw": license_str}

    # Heuristics
    if "nc" in normalized or "non-commercial" in normalized or "noncommercial" in normalized:
        return {"label": "🟡 需审查", "class": "restricted", "raw": license_str}
    if "nd" in normalized or "no-deriv" in normalized:
        return {"label": "🔴 不可商用", "class": "forbidden", "raw": license_str}

    return {"label": "⚪ 需确认", "class": "unknown", "raw": license_str}


# ============================================================
# Model Compatibility
# ============================================================

# Architectures known to work with LLaMA-Factory
SUPPORTED_ARCHITECTURES = {
    "llama", "mistral", "mixtral", "qwen2", "qwen2.5", "gemma", "gemma2",
    "phi", "phi3", "phi4", "falcon", "baichuan", "chatglm", "yi",
    "deepseek", "deepseekv2", "deepseekv3", "internlm", "internlm2",
    "orion", "starcoder2", "codegemma", "command-r", "cohere",
    "dbrx", "mamba", "stablelm", "xverse", "minicpm", "olmo",
    "bloom", "gpt-neox", "gpt2",
}


def check_model_compatibility(model_info: dict) -> dict:
    """
    检查模型是否被 LLaMA-Factory 支持用于微调。

    Returns:
        {
            "supported": bool,
            "architecture": str,
            "reason": str,
        }
    """
    arch = model_info.get("architecture", "") or ""
    model_id = model_info.get("model_id", "").lower()

    # 通过架构名匹配
    for supported in SUPPORTED_ARCHITECTURES:
        if supported in arch.lower():
            return {"supported": True, "architecture": arch, "reason": f"架构 {arch} 受支持"}

    # 通过模型名匹配
    for supported in SUPPORTED_ARCHITECTURES:
        if supported in model_id:
            return {"supported": True, "architecture": arch or supported, "reason": f"模型族匹配"}

    if arch:
        return {
            "supported": False,
            "architecture": arch,
            "reason": f"架构 {arch} 未在已知支持列表中，可能需要手动测试",
        }

    return {
        "supported": False,
        "architecture": "unknown",
        "reason": "无法识别模型架构，建议在 LLaMA-Factory 文档中确认兼容性",
    }


# ============================================================
# Model Search
# ============================================================

@dataclass
class ModelSearchResult:
    """统一的模型搜索结果"""
    model_id: str
    source: str  # "huggingface" | "modelscope"
    author: str = ""
    description: str = ""
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    size_bytes: int | None = None
    size_human: str = ""
    license_info: dict = field(default_factory=dict)
    pipeline_tag: str = ""
    last_modified: str = ""
    compatibility: dict = field(default_factory=dict)
    is_downloaded: bool = False
    local_path: str = ""


def search_huggingface(
    query: str,
    task_type: str | None = None,
    min_params: float | None = None,
    max_params: float | None = None,
    license_filter: str | None = None,
    limit: int = 20,
    sort: str = "downloads",
) -> list[ModelSearchResult]:
    """
    在 Hugging Face Hub 搜索模型。

    Args:
        query: 搜索关键词
        task_type: 任务类型过滤（text-generation, text2text-generation 等）
        min_params/max_params: 参数量范围（B，十亿）
        license_filter: 许可证过滤（commercial / restricted / forbidden / unknown）
        limit: 返回结果数量
        sort: 排序方式（downloads / likes / last_modified）
    """
    results = []

    try:
        api = HfApi()

        # Build filters
        filter_list = []
        if task_type:
            filter_list.append(f"pipeline_tag:{task_type}")
        else:
            # Default: text generation models
            filter_list.append("pipeline_tag:text-generation")

        # Search
        models = list_models(
            search=query,
            limit=limit,
            sort=sort,
            author=None,
            full=False,
            cardData=False,
            fetch_config=False,
        )
    except Exception as e:
        # Fallback: if HF Hub search fails, return empty
        print(f"[ModelHub] HuggingFace search failed: {e}")
        return []

    for model in models:
        if len(results) >= limit:
            break

        try:
            model_id = model.modelId if hasattr(model, 'modelId') else model.id

            # License
            license_str = getattr(model, 'license', None)
            license_info = classify_license(license_str)

            # Filter by license
            if license_filter and license_info["class"] != license_filter:
                continue

            # Extract parameter size from tags or model ID
            tags = getattr(model, 'tags', []) or []
            pipeline_tag = getattr(model, 'pipeline_tag', '') or ''
            downloads = getattr(model, 'downloads', 0) or 0
            likes = getattr(model, 'likes', 0) or 0
            author = getattr(model, 'author', '') or ''
            last_modified = str(getattr(model, 'last_modified', '')) if hasattr(model, 'last_modified') else ''

            result = ModelSearchResult(
                model_id=model_id,
                source="huggingface",
                author=author,
                description="",
                downloads=downloads,
                likes=likes,
                tags=list(tags),
                license_info=license_info,
                pipeline_tag=pipeline_tag,
                last_modified=last_modified,
            )

            # Check compatibility
            result.compatibility = check_model_compatibility({
                "model_id": model_id,
                "architecture": ",".join(tags),
            })

            # Check if downloaded
            result.is_downloaded = is_model_downloaded(model_id)
            if result.is_downloaded:
                result.local_path = str(MODELS_DIR / model_id.replace("/", "--"))

            results.append(result)

        except Exception as e:
            print(f"[ModelHub] Error processing model {getattr(model, 'id', '?')}: {e}")
            continue

    return results


def search_modelscope(
    query: str,
    task_type: str | None = None,
    limit: int = 20,
) -> list[ModelSearchResult]:
    """
    在 ModelScope（魔搭社区）搜索模型。
    """
    results = []

    try:
        from modelscope.hub.api import HubApi
        api = HubApi()

        # ModelScope search
        models = api.list_models(
            query=query,
            limit=limit,
        )
    except ImportError:
        print("[ModelHub] modelscope SDK not installed")
        return []
    except Exception as e:
        print(f"[ModelHub] ModelScope search failed: {e}")
        return []

    for model in models:
        try:
            model_id = model.get("Name", model.get("name", ""))
            if not model_id:
                continue

            license_str = model.get("License", model.get("license", ""))
            license_info = classify_license(license_str)

            tags = model.get("Tags", []) or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",")]

            result = ModelSearchResult(
                model_id=model_id,
                source="modelscope",
                author=model.get("Author", ""),
                description=model.get("Description", ""),
                downloads=model.get("Downloads", 0),
                likes=model.get("Likes", 0),
                tags=tags,
                license_info=license_info,
                pipeline_tag=model.get("Task", ""),
                last_modified=model.get("LastModified", ""),
            )

            result.compatibility = check_model_compatibility({
                "model_id": model_id,
                "architecture": ",".join(tags),
            })

            result.is_downloaded = is_model_downloaded(model_id)
            if result.is_downloaded:
                result.local_path = str(MODELS_DIR / model_id.replace("/", "--"))

            results.append(result)

        except Exception as e:
            print(f"[ModelHub] Error processing ModelScope result: {e}")
            continue

    return results


def unified_search(
    query: str,
    sources: list[str] | None = None,
    task_type: str | None = None,
    license_filter: str | None = None,
    limit: int = 30,
) -> list[ModelSearchResult]:
    """
    统一搜索入口：同时搜索 HF 和 ModelScope，合并结果。

    Args:
        query: 搜索关键词
        sources: 要搜索的源，默认 ["huggingface", "modelscope"]
        task_type: 任务类型
        license_filter: 许可证过滤
        limit: 每个源的结果数量上限
    """
    if sources is None:
        sources = ["huggingface", "modelscope"]

    all_results = []

    if "huggingface" in sources:
        hf_limit = limit if "modelscope" not in sources else limit // 2 + 1
        hf_results = search_huggingface(
            query=query,
            task_type=task_type,
            license_filter=license_filter,
            limit=hf_limit,
        )
        all_results.extend(hf_results)

    if "modelscope" in sources:
        ms_limit = limit if "huggingface" not in sources else limit // 2 + 1
        ms_results = search_modelscope(
            query=query,
            task_type=task_type,
            limit=ms_limit,
        )
        all_results.extend(ms_results)

    # Sort by downloads descending
    all_results.sort(key=lambda x: x.downloads, reverse=True)

    return all_results[:limit]


# ============================================================
# Model Download with Resume Support
# ============================================================

class ModelDownloader:
    """
    支持断点续传的模型下载器。

    使用 HuggingFace Hub 的 hf_hub_download，它原生支持断点续传。
    """

    def __init__(self, model_id: str, save_dir: Path | None = None):
        self.model_id = model_id
        self.save_dir = save_dir or (MODELS_DIR / model_id.replace("/", "--"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._cancelled = False

    @property
    def progress_file(self) -> Path:
        return self.save_dir / ".download_progress.json"

    def cancel(self):
        """取消下载"""
        self._cancelled = True

    def get_download_status(self) -> dict:
        """获取当前下载状态"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"status": "not_started", "downloaded_files": [], "total_size_mb": 0}

    def download(
        self,
        progress_callback=None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> dict:
        """
        下载模型文件。

        Args:
            progress_callback: 进度回调函数 callback(current, total, filename)
            allow_patterns: 只下载匹配的文件（如 ["*.safetensors", "*.json"]）
            ignore_patterns: 忽略匹配的文件（如 ["*.bin", "*.msgpack"]）

        Returns:
            {
                "success": bool,
                "path": str,
                "files": list of downloaded files,
                "total_size_mb": float,
                "error": str (if failed),
            }
        """
        if ignore_patterns is None:
            ignore_patterns = ["*.pth", "*.bin", "*.msgpack", "*.h5", "*.ot"]

        downloaded_files = []
        total_size_mb = 0.0

        try:
            # First, get the list of files to download
            api = HfApi()
            try:
                repo_files = api.list_repo_files(self.model_id)
            except Exception:
                repo_files = []

            # Filter files
            files_to_download = []
            for f in repo_files:
                # Apply ignore patterns
                if ignore_patterns:
                    if any(_match_pattern(f, p) for p in ignore_patterns):
                        continue
                # Apply allow patterns
                if allow_patterns:
                    if not any(_match_pattern(f, p) for p in allow_patterns):
                        continue
                files_to_download.append(f)

            if not files_to_download:
                # Fallback: download common model files
                files_to_download = None  # hf_hub_download will download all

            # Download each file
            for i, filename in enumerate(files_to_download or ["*"]):
                if self._cancelled:
                    return {
                        "success": False,
                        "path": str(self.save_dir),
                        "files": downloaded_files,
                        "total_size_mb": total_size_mb,
                        "error": "下载已取消",
                    }

                try:
                    local_path = hf_hub_download(
                        repo_id=self.model_id,
                        filename=filename if filename != "*" else None,
                        local_dir=self.save_dir,
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                    downloaded_files.append(local_path)

                    if os.path.exists(local_path):
                        size_mb = os.path.getsize(local_path) / (1024 ** 2)
                        total_size_mb += size_mb

                        if progress_callback:
                            progress_callback(
                                current=i + 1,
                                total=len(files_to_download) if files_to_download else 1,
                                filename=os.path.basename(local_path),
                                size_mb=size_mb,
                            )

                except Exception as e:
                    print(f"[ModelHub] Failed to download {filename}: {e}")
                    continue

            # Save progress
            progress_data = {
                "status": "completed",
                "model_id": self.model_id,
                "downloaded_files": downloaded_files,
                "total_size_mb": round(total_size_mb, 2),
                "completed_at": time.time(),
            }
            with open(self.progress_file, "w") as f:
                json.dump(progress_data, f, indent=2)

            return {
                "success": True,
                "path": str(self.save_dir),
                "files": downloaded_files,
                "total_size_mb": round(total_size_mb, 2),
                "error": "",
            }

        except HfHubHTTPError as e:
            return {
                "success": False,
                "path": str(self.save_dir),
                "files": downloaded_files,
                "total_size_mb": total_size_mb,
                "error": f"HTTP 错误: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "path": str(self.save_dir),
                "files": downloaded_files,
                "total_size_mb": total_size_mb,
                "error": f"下载失败: {e}",
            }


def _match_pattern(filename: str, pattern: str) -> bool:
    """简单的 glob 模式匹配"""
    import fnmatch
    return fnmatch.fnmatch(os.path.basename(filename), pattern)


# ============================================================
# Local Model Library
# ============================================================

def is_model_downloaded(model_id: str) -> bool:
    """检查模型是否已下载到本地"""
    model_dir = MODELS_DIR / model_id.replace("/", "--")
    if not model_dir.exists():
        return False
    # Check for model files
    model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.gguf"))
    config_file = model_dir / "config.json"
    return len(model_files) > 0 or config_file.exists()


def get_local_model_info(model_id: str) -> dict | None:
    """获取本地模型的详细信息"""
    model_dir = MODELS_DIR / model_id.replace("/", "--")
    if not model_dir.exists():
        return None

    # Calculate size
    total_size = 0
    file_count = 0
    for f in model_dir.rglob("*"):
        if f.is_file() and not f.name.startswith("."):
            total_size += f.stat().st_size
            file_count += 1

    # Read config if exists
    config = {}
    config_file = model_dir / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
        except Exception:
            pass

    return {
        "model_id": model_id,
        "local_path": str(model_dir),
        "size_gb": round(total_size / (1024 ** 3), 2),
        "file_count": file_count,
        "architecture": config.get("architectures", []),
        "model_type": config.get("model_type", ""),
        "hidden_size": config.get("hidden_size", 0),
        "num_layers": config.get("num_hidden_layers", 0),
        "vocab_size": config.get("vocab_size", 0),
    }


def list_local_models() -> list[dict]:
    """列出所有本地已下载的模型"""
    models = []

    if not MODELS_DIR.exists():
        return models

    for model_dir in MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        # Reverse the directory name conversion
        model_id = model_dir.name.replace("--", "/")
        info = get_local_model_info(model_id)
        if info:
            # Add license info from config
            config_file = model_dir / "config.json"
            license_str = None
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                    license_str = config.get("license", None)
                except Exception:
                    pass
            info["license_info"] = classify_license(license_str)
            models.append(info)

    return models


def delete_local_model(model_id: str) -> dict:
    """删除本地模型"""
    model_dir = MODELS_DIR / model_id.replace("/", "--")
    import shutil

    if not model_dir.exists():
        return {"success": False, "error": "模型目录不存在"}

    try:
        size_before = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        shutil.rmtree(model_dir)
        return {
            "success": True,
            "freed_space_gb": round(size_before / (1024 ** 3), 2),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_disk_usage_for_models() -> dict:
    """获取模型存储目录的磁盘使用情况"""
    if not MODELS_DIR.exists():
        return {"model_count": 0, "total_size_gb": 0.0, "models": []}

    models = list_local_models()
    total_size = sum(m.get("size_gb", 0) for m in models)

    return {
        "model_count": len(models),
        "total_size_gb": round(total_size, 2),
        "models": models,
    }


# ============================================================
# Model Details & Recommendations
# ============================================================

MODEL_RECOMMENDATIONS = [
    {
        "name": "Qwen2.5-7B-Instruct",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "source": "huggingface",
        "params_b": 7.0,
        "size_gb": 14.5,
        "description": "阿里通义千问2.5，中文表现优秀，7B 参数适合消费级显卡",
        "tags": ["对话", "中文", "指令微调"],
        "license": "🟢 Apache 2.0",
        "min_vram": {"qlora": 6, "lora": 16, "full": 40},
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "source": "huggingface",
        "params_b": 8.0,
        "size_gb": 16.0,
        "description": "Meta 最新开源模型，综合能力强",
        "tags": ["对话", "英文/多语言", "指令微调"],
        "license": "🟢 Llama 3.1 Community",
        "min_vram": {"qlora": 6, "lora": 16, "full": 40},
    },
    {
        "name": "Gemma-2-2B-Instruct",
        "model_id": "google/gemma-2-2b-it",
        "source": "huggingface",
        "params_b": 2.0,
        "size_gb": 4.0,
        "description": "Google 轻量模型，适合小显存入门学习",
        "tags": ["对话", "轻量", "入门"],
        "license": "🟢 Gemma",
        "min_vram": {"qlora": 4, "lora": 8, "full": 12},
    },
    {
        "name": "DeepSeek-Coder-6.7B",
        "model_id": "deepseek-ai/DeepSeek-Coder-6.7B-Instruct",
        "source": "huggingface",
        "params_b": 6.7,
        "size_gb": 13.4,
        "description": "深度求索代码模型，擅长编程任务",
        "tags": ["代码", "编程", "指令微调"],
        "license": "🟢 DeepSeek",
        "min_vram": {"qlora": 6, "lora": 14, "full": 36},
    },
    {
        "name": "Qwen2.5-1.5B-Instruct",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "source": "huggingface",
        "params_b": 1.5,
        "size_gb": 3.0,
        "description": "超轻量中文模型，入门首选",
        "tags": ["对话", "轻量", "入门", "中文"],
        "license": "🟢 Apache 2.0",
        "min_vram": {"qlora": 4, "lora": 6, "full": 10},
    },
    {
        "name": "Yi-1.5-6B-Chat",
        "model_id": "01-ai/Yi-1.5-6B-Chat",
        "source": "huggingface",
        "params_b": 6.0,
        "size_gb": 12.0,
        "description": "零一万物中英双语模型",
        "tags": ["对话", "中英双语", "指令微调"],
        "license": "🟢 Apache 2.0",
        "min_vram": {"qlora": 6, "lora": 14, "full": 32},
    },
]


def get_model_detail(model_id: str) -> dict | None:
    """
    获取模型的详细信息（用于展示详情页）。

    优先从本地获取，其次从 HF API 获取。
    """
    # Check local first
    local_info = get_local_model_info(model_id)

    # Try HF API for remote info
    remote_info = {}
    try:
        api = HfApi()
        model_info = api.model_info(model_id)

        remote_info = {
            "model_id": model_id,
            "author": getattr(model_info, 'author', ''),
            "description": getattr(model_info, 'description', '') or '',
            "downloads": getattr(model_info, 'downloads', 0),
            "likes": getattr(model_info, 'likes', 0),
            "tags": list(getattr(model_info, 'tags', []) or []),
            "pipeline_tag": getattr(model_info, 'pipeline_tag', ''),
            "license": getattr(model_info, 'license', ''),
            "last_modified": str(getattr(model_info, 'last_modified', '')) if hasattr(model_info, 'last_modified') else '',
        }
    except Exception:
        pass

    if local_info:
        return {**remote_info, **local_info, "is_downloaded": True}

    if remote_info:
        remote_info["is_downloaded"] = False
        remote_info["license_info"] = classify_license(remote_info.get("license"))
        return remote_info

    return None


def get_model_card(model_id: str) -> str:
    """
    获取模型的 README / Model Card 内容。
    """
    try:
        api = HfApi()
        # Try to get the model card (README.md)
        try:
            card = api.model_info(model_id).card_data
            if card:
                return str(card)
        except Exception:
            pass
    except Exception:
        pass

    # Fallback: try to read local README
    model_dir = MODELS_DIR / model_id.replace("/", "--")
    for readme_name in ["README.md", "readme.md", "model_card.md"]:
        readme_path = model_dir / readme_name
        if readme_path.exists():
            return readme_path.read_text(encoding='utf-8', errors='ignore')

    return ""


# ============================================================
# Enhanced Search with Cache & Offline Fallback
# ============================================================

SEARCH_CACHE_FILE = MODELS_DIR.parent / "search_cache.json"
SEARCH_CACHE: dict = {}
_SEARCH_CACHE_DIRTY = False


def _load_search_cache():
    """加载搜索缓存"""
    global SEARCH_CACHE
    if SEARCH_CACHE:
        return
    try:
        if SEARCH_CACHE_FILE.exists():
            with open(SEARCH_CACHE_FILE) as f:
                SEARCH_CACHE = json.load(f)
    except Exception:
        SEARCH_CACHE = {}


def _save_search_cache():
    """保存搜索缓存"""
    global _SEARCH_CACHE_DIRTY
    if not _SEARCH_CACHE_DIRTY:
        return
    try:
        SEARCH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SEARCH_CACHE_FILE, "w") as f:
            json.dump(SEARCH_CACHE, f, ensure_ascii=False)
        _SEARCH_CACHE_DIRTY = False
    except Exception:
        pass


def enhanced_search(
    query: str,
    sources: list[str] | None = None,
    task_type: str = "",
    license_filter: str = "",
    limit: int = 20,
    use_cache: bool = True,
) -> dict:
    """
    增强版模型搜索：缓存 + 离线降级 + 卡片集成。

    Returns:
        {
            "results": list of dict,
            "source": "online" | "cache" | "offline",
            "total": int,
            "query_time_ms": float,
            "error": str (if any),
        }
    """
    import time
    start = time.time()
    _load_search_cache()

    cache_key = f"{query}|{task_type}|{license_filter}|{'-'.join(sources or ['hf','ms'])}"

    # Try online search first
    online_ok = False
    enriched_results = []
    error_msg = ""

    try:
        raw_results = unified_search(
            query=query, sources=sources,
            task_type=task_type or None,
            license_filter=license_filter or None,
            limit=limit,
        )

        if raw_results:
            online_ok = True
            # Cache results
            SEARCH_CACHE[cache_key] = {
                "timestamp": time.time(),
                "results": [
                    {
                        "model_id": r.model_id,
                        "source": r.source,
                        "author": r.author,
                        "downloads": r.downloads,
                        "likes": r.likes,
                        "license_label": r.license_info.get("label", ""),
                        "license_class": r.license_info.get("class", ""),
                        "compatible": r.compatibility.get("supported", False),
                        "is_downloaded": r.is_downloaded,
                    }
                    for r in raw_results
                ],
            }
            _SEARCH_CACHE_DIRTY = True
            _save_search_cache()
    except Exception as e:
        error_msg = str(e)[:200]

    # Enrich with model cards
    from .model_cards import match_card
    source_results = raw_results if online_ok else []

    # If offline or no results, use cache
    if not online_ok and use_cache:
        cached = SEARCH_CACHE.get(cache_key, {})
        if cached and cached.get("results"):
            source_results = cached["results"]
            error_msg = f"离线模式：使用缓存结果（{len(source_results)}条）" if error_msg else ""

    # If still no results, show local models
    if not source_results:
        local_models = list_local_models()
        if local_models:
            enriched_results = [
                {
                    "model_id": m["model_id"],
                    "source": "local",
                    "author": "",
                    "downloads": 0,
                    "license_label": m.get("license_info", {}).get("label", "⚪ 未知"),
                    "license_class": "unknown",
                    "compatible": True,
                    "is_downloaded": True,
                    "card_match": None,
                }
                for m in local_models
            ]
            error_msg = "离线模式：仅显示本地模型"

    # Enrich each result with card info
    for r in source_results:
        result_dict = r if isinstance(r, dict) else {
            "model_id": r.model_id, "source": r.source, "author": r.author,
            "downloads": r.downloads, "license_label": r.license_info.get("label", ""),
            "license_class": r.license_info.get("class", ""),
            "compatible": r.compatibility.get("supported", False),
            "is_downloaded": r.is_downloaded,
        }
        card = match_card(result_dict["model_id"])
        result_dict["card_match"] = {
            "display_name": card["display_name"],
            "params_b": card["params_b"],
            "family": card["family"],
            "min_vram_qlora": card["training"]["qlora"]["min_vram_gb"],
        } if card else None
        enriched_results.append(result_dict)

    elapsed = (time.time() - start) * 1000

    return {
        "results": enriched_results[:limit],
        "source": "online" if online_ok else ("cache" if source_results else "offline"),
        "total": len(enriched_results),
        "query_time_ms": round(elapsed, 1),
        "error": error_msg,
    }


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    # Test search
    print("=" * 60)
    print("🔍 搜索 Qwen 模型...")
    print("=" * 60)

    results = unified_search("Qwen 7B", sources=["huggingface"], limit=5)

    for r in results:
        lic = r.license_info.get("label", "?")
        comp = "✅" if r.compatibility.get("supported") else "❓"
        print(f"  {comp} {r.model_id}")
        print(f"     {lic} · 📥 {r.downloads:,} · 已下载: {r.is_downloaded}")

    print()
    print("=" * 60)
    print("📦 本地模型库")
    print("=" * 60)
    local = list_local_models()
    if local:
        for m in local:
            print(f"  📦 {m['model_id']} ({m['size_gb']} GB)")
    else:
        print("  (空)")

    print()
    print("=" * 60)
    print("🎯 推荐模型")
    print("=" * 60)
    for rec in MODEL_RECOMMENDATIONS[:3]:
        print(f"  ⭐ {rec['name']}")
        print(f"     {rec['description']}")
        print(f"     {rec['license']} · {rec['size_gb']}GB · 最低显存: {rec['min_vram']['qlora']}GB (QLoRA)")
