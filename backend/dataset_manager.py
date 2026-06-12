"""
数据集管理：内置库、在线搜索、数据生成向导、格式转换、清洗、预览

这是模型训练室的核心竞争力模块。
"""

import json
import os
import random
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# ============================================================
# Configuration
# ============================================================

DATASETS_DIR = Path(__file__).parent.parent / "data" / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Data Classes
# ============================================================

@dataclass
class DatasetInfo:
    """数据集的统一描述"""
    id: str
    name: str
    source: str  # "builtin" | "huggingface" | "modelscope" | "local"
    category: str  # 对话 / 代码 / 翻译 / 客服 / 安全 / 自定义
    format: str  # "sharegpt" | "alpaca" | "conversation" | "raw"
    record_count: int = 0
    size_mb: float = 0.0
    license_info: dict = field(default_factory=dict)
    description: str = ""
    sample_data: list = field(default_factory=list)  # 前 3 条预览
    tags: list[str] = field(default_factory=list)
    local_path: str = ""
    train_split: float = 0.9


# ============================================================
# Built-in Dataset Library
# ============================================================

BUILTIN_DATASETS = [
    {
        "id": "alpaca-zh",
        "name": "Chinese-Alpaca (中文指令)",
        "source": "builtin",
        "category": "对话",
        "format": "alpaca",
        "record_count": 50000,
        "size_mb": 12.0,
        "license": "🟢 MIT",
        "description": "中文指令微调数据集，覆盖写作、问答、翻译等多种任务。适合训练通用中文助手。",
        "tags": ["中文", "指令微调", "通用", "入门"],
        "sample": [
            {"instruction": "介绍一下中国的四大发明", "output": "中国古代四大发明包括指南针、火药、造纸术和印刷术。这些发明对世界文明进程产生了深远影响：1）指南针推动了航海探险...", "length": 234},
            {"instruction": "写一首关于春天的五言绝句", "output": "春风拂面来，桃花朵朵开。燕子衔泥至，新绿满窗台。", "length": 156},
            {"instruction": "解释什么是机器学习，用通俗的语言", "output": "机器学习是人工智能的一个分支。简单来说，就是让计算机通过看大量的数据来自己学习规律，而不是人工一条一条写规则...", "length": 412},
        ],
    },
    {
        "id": "code-alpaca",
        "name": "Code-Alpaca (代码助手)",
        "source": "builtin",
        "category": "代码",
        "format": "alpaca",
        "record_count": 20000,
        "size_mb": 5.0,
        "license": "🟢 Apache 2.0",
        "description": "代码生成与解释数据集，使用 Self-Instruct 从 ChatGPT 生成。适合训练代码助手。",
        "tags": ["代码", "编程", "英文"],
        "sample": [
            {"instruction": "用 Python 写一个二分查找函数", "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1", "length": 350},
            {"instruction": "解释什么是递归，并给出示例", "output": "递归是一种编程技巧，指函数调用自身来解决问题。它通常包含两个部分：基线条件（停止条件）和递归条件...", "length": 280},
        ],
    },
    {
        "id": "belle-zh",
        "name": "BELLE (中文对话)",
        "source": "builtin",
        "category": "对话",
        "format": "conversation",
        "record_count": 100000,
        "size_mb": 25.0,
        "license": "🟢 Apache 2.0",
        "description": "链家 BELLE 项目发布的中文对话数据集，基于 ChatGPT 生成，质量高。适合训练中文聊天模型。",
        "tags": ["中文", "对话", "高质量"],
        "sample": [
            {"conversations": [
                {"from": "human", "value": "你好，今天天气真好"},
                {"from": "assistant", "value": "是的呢！阳光明媚，适合出去走走。你今天有什么计划吗？"}
            ], "length": 145},
        ],
    },
    {
        "id": "firefly-zh",
        "name": "Firefly (中文多任务)",
        "source": "builtin",
        "category": "对话",
        "format": "alpaca",
        "record_count": 60000,
        "size_mb": 15.0,
        "license": "🟢 MIT",
        "description": "Firefly 项目的中文数据集，涵盖 23 种 NLP 任务。适合训练多任务中文模型。",
        "tags": ["中文", "多任务", "通用"],
        "sample": [],
    },
    {
        "id": "ecs-zh",
        "name": "电商客服对话 (中文)",
        "source": "builtin",
        "category": "客服",
        "format": "conversation",
        "record_count": 12000,
        "size_mb": 3.0,
        "license": "🟢 CC-BY",
        "description": "电商场景下的真实客服对话，含售前咨询、售后问题等。适合训练客服机器人。",
        "tags": ["中文", "客服", "电商", "对话"],
        "sample": [
            {"conversations": [
                {"from": "human", "value": "我的订单怎么还没到？都已经三天了"},
                {"from": "assistant", "value": "非常抱歉给您带来不便！请您提供一下订单号，我马上帮您查询物流状态。"}
            ], "length": 98},
        ],
    },
    {
        "id": "wmt-zh-en",
        "name": "WMT 中英翻译子集",
        "source": "builtin",
        "category": "翻译",
        "format": "alpaca",
        "record_count": 100000,
        "size_mb": 25.0,
        "license": "🟡 学术用途",
        "description": "WMT 机器翻译比赛数据的中英子集，适合训练翻译模型。注意：仅限学术用途。",
        "tags": ["翻译", "中英", "学术"],
        "sample": [
            {"instruction": "将以下中文翻译成英文：人工智能正在改变世界", "output": "Artificial intelligence is changing the world.", "length": 120},
        ],
    },
    {
        "id": "dolly-zh",
        "name": "Databricks-Dolly (中文版)",
        "source": "builtin",
        "category": "对话",
        "format": "alpaca",
        "record_count": 15000,
        "size_mb": 4.0,
        "license": "🟢 CC-BY-SA",
        "description": "Databricks Dolly 15K 的中文翻译版。涵盖头脑风暴、分类、封闭式QA、生成、信息提取、开放式QA和总结等类别。",
        "tags": ["中文", "指令微调", "入门"],
        "sample": [],
    },
    {
        "id": "gcq-zh",
        "name": "通用客服问答 (中文)",
        "source": "builtin",
        "category": "客服",
        "format": "alpaca",
        "record_count": 8000,
        "size_mb": 2.0,
        "license": "🟢 MIT",
        "description": "通用客服场景问答对，覆盖退换货、投诉、产品咨询等场景。",
        "tags": ["中文", "客服", "小数据集"],
        "sample": [],
    },
    {
        "id": "hh-rlhf-zh",
        "name": "HH-RLHF 安全对齐子集",
        "source": "builtin",
        "category": "安全",
        "format": "conversation",
        "record_count": 10000,
        "size_mb": 3.0,
        "license": "🟢 MIT",
        "description": "人类偏好数据的中文子集，用于RLHF安全对齐。帮助模型学会拒绝有害请求。",
        "tags": ["安全", "对齐", "偏好"],
        "sample": [],
    },
    {
        "id": "sharegpt-zh",
        "name": "ShareGPT 中文对话",
        "source": "builtin",
        "category": "对话",
        "format": "sharegpt",
        "record_count": 80000,
        "size_mb": 40.0,
        "license": "🟢 Apache 2.0",
        "description": "ShareGPT 用户分享的 ChatGPT 对话中文子集。多轮对话质量高，非常适合训练对话模型。",
        "tags": ["中文", "多轮对话", "高质量"],
        "sample": [],
    },
]


def get_builtin_dataset(dataset_id: str) -> dict | None:
    """获取内置数据集的详细信息"""
    for ds in BUILTIN_DATASETS:
        if ds["id"] == dataset_id:
            return ds.copy()
    return None


def list_builtin_datasets(category: str | None = None) -> list[dict]:
    """列出所有内置数据集，可按分类过滤"""
    if category:
        return [ds.copy() for ds in BUILTIN_DATASETS if ds["category"] == category]
    return [ds.copy() for ds in BUILTIN_DATASETS]


# ============================================================
# Format Conversion
# ============================================================

def detect_format(data: list[dict]) -> str:
    """
    自动检测数据格式。

    Supported formats:
    - "sharegpt": {"messages": [{"role": "user", "content": "..."}, ...]}
    - "alpaca": {"instruction": "...", "input": "...", "output": "..."}
    - "conversation": {"conversations": [{"from": "human", "value": "..."}, ...]}
    - "json": raw JSON
    """
    if not data:
        return "unknown"

    sample = data[0]
    keys = set(sample.keys()) if isinstance(sample, dict) else set()

    if "messages" in keys and isinstance(sample.get("messages"), list):
        return "sharegpt"
    if "conversations" in keys and isinstance(sample.get("conversations"), list):
        return "conversation"
    if "instruction" in keys and "output" in keys:
        return "alpaca"
    if "input" in keys and "output" in keys:
        return "alpaca"  # input 可为空
    if "prompt" in keys and "response" in keys:
        return "alpaca"
    if "question" in keys and "answer" in keys:
        return "alpaca"

    return "raw"


def convert_to_sharegpt(data: list[dict], source_format: str | None = None) -> list[dict]:
    """
    将各种格式统一转换为 ShareGPT 格式。

    ShareGPT format:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    if source_format is None:
        source_format = detect_format(data)

    if source_format == "sharegpt":
        return data

    converted = []

    for item in data:
        messages = []

        if source_format == "alpaca":
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            # Combine instruction and input
            if input_text:
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction

            if user_content:
                messages.append({"role": "user", "content": user_content})
            if output:
                messages.append({"role": "assistant", "content": output})

        elif source_format == "conversation":
            convs = item.get("conversations", [])
            for turn in convs:
                role_map = {
                    "human": "user",
                    "user": "user",
                    "assistant": "assistant",
                    "gpt": "assistant",
                    "bot": "assistant",
                    "from": "user",  # fallback
                }
                from_field = turn.get("from", turn.get("role", ""))
                value = turn.get("value", turn.get("content", ""))
                role = role_map.get(from_field, "user")
                messages.append({"role": role, "content": value})

        elif source_format == "raw":
            # Try common field pairs
            question = item.get("question", item.get("prompt", item.get("query", "")))
            answer = item.get("answer", item.get("response", item.get("reply", "")))
            if question:
                messages.append({"role": "user", "content": question})
            if answer:
                messages.append({"role": "assistant", "content": answer})

        if messages:
            converted.append({"messages": messages})

    return converted


def convert_to_alpaca(data: list[dict], source_format: str | None = None) -> list[dict]:
    """
    将各种格式统一转换为 Alpaca 格式。

    Alpaca format:
    {
        "instruction": "...",
        "input": "",
        "output": "..."
    }
    """
    if source_format is None:
        source_format = detect_format(data)

    if source_format == "alpaca":
        return data

    converted = []

    for item in data:
        if source_format == "sharegpt":
            messages = item.get("messages", [])
            instruction = ""
            output = ""
            for msg in messages:
                if msg["role"] == "user":
                    instruction = msg["content"]
                elif msg["role"] == "assistant":
                    output = msg["content"]
            if instruction and output:
                converted.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                })

        elif source_format == "conversation":
            convs = item.get("conversations", [])
            instruction = ""
            output = ""
            for turn in convs:
                from_field = turn.get("from", turn.get("role", ""))
                value = turn.get("value", turn.get("content", ""))
                if from_field in ("human", "user"):
                    instruction = value
                elif from_field in ("assistant", "gpt", "bot"):
                    output = value
            if instruction and output:
                converted.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                })

    return converted


# ============================================================
# Data Import
# ============================================================

def import_file(file_path: str) -> dict:
    """
    导入数据文件（JSON/CSV/TXT），自动检测格式并转换。

    Returns:
        {
            "success": bool,
            "data": list of converted records,
            "format": detected format,
            "record_count": int,
            "error": str (if failed),
        }
    """
    path = Path(file_path)

    if not path.exists():
        return {"success": False, "data": [], "format": "", "record_count": 0, "error": "文件不存在"}

    try:
        ext = path.suffix.lower()

        if ext == ".json" or ext == ".jsonl":
            data = _load_json(path)
        elif ext == ".csv":
            data = _load_csv(path)
        elif ext == ".parquet":
            data = _load_parquet(path)
        elif ext == ".txt":
            data = _load_txt(path)
        else:
            return {"success": False, "data": [], "format": "", "record_count": 0,
                    "error": f"不支持的文件格式: {ext}"}

        if not data:
            return {"success": False, "data": [], "format": "", "record_count": 0,
                    "error": "文件中未找到有效数据"}

        fmt = detect_format(data)
        converted = convert_to_sharegpt(data, source_format=fmt)

        return {
            "success": True,
            "data": converted,
            "format": fmt,
            "record_count": len(converted),
            "error": "",
        }

    except Exception as e:
        return {"success": False, "data": [], "format": "", "record_count": 0,
                "error": f"导入失败: {str(e)}"}


def _load_json(path: Path) -> list[dict]:
    """加载 JSON 文件"""
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    # Try JSON Lines
    if content.startswith("{"):
        lines = content.splitlines()
        data = []
        for line in lines:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if data:
            return data
    # Try regular JSON
    parsed = json.loads(content)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # Try common keys
        for key in ("data", "items", "examples", "records", "rows"):
            if key in parsed:
                return parsed[key]
        return [parsed]
    return []


def _load_csv(path: Path) -> list[dict]:
    """加载 CSV 文件"""
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _load_parquet(path: Path) -> list[dict]:
    """加载 Parquet 文件"""
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def _load_txt(path: Path) -> list[dict]:
    """加载 TXT 文件（每行一个问答对，用 tab 分隔）"""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                data.append({"question": parts[0], "answer": parts[1]})
            elif len(parts) == 1:
                data.append({"instruction": parts[0], "output": ""})
    return data


# ============================================================
# Data Cleaning
# ============================================================

def clean_dataset(data: list[dict], options: dict | None = None) -> dict:
    """
    数据清洗。

    Args:
        data: ShareGPT 格式的数据
        options: {
            "remove_duplicates": bool (default True),
            "remove_empty": bool (default True),
            "min_length": int (default 10),
            "max_length": int (default 8192),
            "trim_whitespace": bool (default True),
        }

    Returns:
        {
            "cleaned_data": list,
            "stats": {removed_count, duplicate_count, empty_count, too_short, too_long}
        }
    """
    if options is None:
        options = {}

    remove_duplicates = options.get("remove_duplicates", True)
    remove_empty = options.get("remove_empty", True)
    min_len = options.get("min_length", 10)
    max_len = options.get("max_length", 8192)
    trim_ws = options.get("trim_whitespace", True)

    stats = {"total": len(data), "removed": 0, "duplicate": 0, "empty": 0, "too_short": 0, "too_long": 0}
    cleaned = []

    seen = set()

    for item in data:
        messages = item.get("messages", [])
        if not messages:
            stats["empty"] += 1
            continue

        # Check empty
        if remove_empty:
            has_content = any(m.get("content", "").strip() for m in messages)
            if not has_content:
                stats["empty"] += 1
                continue

        # Trim whitespace
        if trim_ws:
            for m in messages:
                m["content"] = m["content"].strip()

        # Check length
        total_len = sum(len(m.get("content", "")) for m in messages)
        if total_len < min_len:
            stats["too_short"] += 1
            continue
        if total_len > max_len:
            stats["too_long"] += 1
            continue

        # Check duplicates
        if remove_duplicates:
            fingerprint = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            if fingerprint in seen:
                stats["duplicate"] += 1
                continue
            seen.add(fingerprint)

        cleaned.append(item)

    stats["removed"] = stats["total"] - len(cleaned)

    return {"cleaned_data": cleaned, "stats": stats}


# ============================================================
# Data Preview
# ============================================================

def preview_data(data: list[dict], sample_count: int = 5) -> list[dict]:
    """返回数据样本用于预览"""
    if len(data) <= sample_count:
        return data
    # Take evenly distributed samples
    indices = [int(i * len(data) / sample_count) for i in range(sample_count)]
    return [data[i] for i in indices]


def get_data_stats(data: list[dict]) -> dict:
    """
    生成数据质量报告。

    Returns:
        {
            "total_count": int,
            "avg_user_length": float,
            "avg_assistant_length": float,
            "length_distribution": {bucket: count},
            "min_length": int,
            "max_length": int,
            "language_hint": str (zh/en/mixed),
        }
    """
    if not data:
        return {"total_count": 0}

    user_lengths = []
    assistant_lengths = []
    all_text = []
    total_lengths = []

    for item in data:
        messages = item.get("messages", [])
        item_text = []
        for m in messages:
            content = m.get("content", "")
            if m.get("role") == "user":
                user_lengths.append(len(content))
            elif m.get("role") == "assistant":
                assistant_lengths.append(len(content))
            item_text.append(content)
        all_text.extend(item_text)
        total_lengths.append(sum(len(t) for t in item_text))

    # Length distribution buckets
    buckets = {"0-50": 0, "50-200": 0, "200-500": 0, "500-1000": 0, "1000-2000": 0, "2000+": 0}
    for l in total_lengths:
        if l < 50:
            buckets["0-50"] += 1
        elif l < 200:
            buckets["50-200"] += 1
        elif l < 500:
            buckets["200-500"] += 1
        elif l < 1000:
            buckets["500-1000"] += 1
        elif l < 2000:
            buckets["1000-2000"] += 1
        else:
            buckets["2000+"] += 1

    # Language detection (simple heuristic)
    all_text_combined = "".join(all_text[:1000])
    chinese_chars = sum(1 for c in all_text_combined if '一' <= c <= '鿿')
    if chinese_chars / max(len(all_text_combined), 1) > 0.3:
        language = "zh"
    elif chinese_chars / max(len(all_text_combined), 1) > 0.05:
        language = "mixed"
    else:
        language = "en"

    return {
        "total_count": len(data),
        "avg_user_length": round(sum(user_lengths) / max(len(user_lengths), 1), 1),
        "avg_assistant_length": round(sum(assistant_lengths) / max(len(assistant_lengths), 1), 1),
        "length_distribution": buckets,
        "min_total_length": min(total_lengths) if total_lengths else 0,
        "max_total_length": max(total_lengths) if total_lengths else 0,
        "avg_total_length": round(sum(total_lengths) / max(len(total_lengths), 1), 1),
        "language_hint": language,
    }


# ============================================================
# Train/Validation Split
# ============================================================

def split_dataset(data: list[dict], train_ratio: float = 0.9, shuffle: bool = True,
                  stratify: bool = False) -> dict:
    """
    切分训练集和验证集。

    Returns:
        {"train": list, "val": list, "train_count": int, "val_count": int}
    """
    if shuffle:
        random.shuffle(data)

    split_idx = int(len(data) * train_ratio)

    return {
        "train": data[:split_idx],
        "val": data[split_idx:],
        "train_count": split_idx,
        "val_count": len(data) - split_idx,
    }


# ============================================================
# Data Generation Wizard (Method A: Few-shot → Batch expansion)
# ============================================================

FEWSHOT_PROMPT_TEMPLATE = """你是一个数据标注专家。我需要你根据下面的示例，生成更多类似格式的训练数据。

## 任务场景
{scenario}

## 示例数据
{examples}

## 要求
1. 严格按照示例的格式和风格
2. 覆盖不同难度和主题
3. 答案要准确、详细、有帮助
4. 可以适当变化提问方式
5. 生成 {count} 条新数据

请直接输出 JSON 数组，每项格式与示例一致。不要输出解释。
"""


def generate_examples_from_fewshot(
    scenario: str,
    examples: list[dict],
    target_count: int = 100,
    llm_api_func=None,
) -> dict:
    """
    基于少量示例，用大模型批量扩充训练数据。

    Args:
        scenario: 任务场景描述
        examples: 3-5 个示例数据
        target_count: 目标生成数量
        llm_api_func: 大模型 API 调用函数，签名为 func(prompt: str) -> str

    Returns:
        {
            "success": bool,
            "generated_data": list,
            "count": int,
            "error": str,
        }
    """
    if llm_api_func is None:
        return {
            "success": False,
            "generated_data": [],
            "count": 0,
            "error": "未配置大模型 API。请在设置中配置 API Key。",
        }

    # Format examples
    examples_text = json.dumps(examples, ensure_ascii=False, indent=2)

    all_generated = []
    batch_size = min(20, target_count)
    remaining = target_count

    while remaining > 0:
        prompt = FEWSHOT_PROMPT_TEMPLATE.format(
            scenario=scenario,
            examples=examples_text,
            count=min(batch_size, remaining),
        )

        try:
            response = llm_api_func(prompt)
            # Try to parse JSON
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                # Extract JSON from code block
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            batch_data = json.loads(response)
            if isinstance(batch_data, list):
                all_generated.extend(batch_data)
                remaining -= len(batch_data)
            else:
                all_generated.append(batch_data)
                remaining -= 1

        except Exception as e:
            return {
                "success": False,
                "generated_data": all_generated,
                "count": len(all_generated),
                "error": f"生成过程出错: {str(e)}",
            }

    # Convert to ShareGPT format
    converted = convert_to_sharegpt(all_generated)

    return {
        "success": True,
        "generated_data": converted,
        "count": len(converted),
        "error": "",
    }


# ============================================================
# Dataset Save & Load
# ============================================================

def save_dataset(data: list[dict], name: str, format: str = "sharegpt") -> str:
    """
    保存数据集到本地。

    Returns:
        path to saved file
    """
    dataset_dir = DATASETS_DIR / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    file_path = dataset_dir / "data.json"

    # If format is not sharegpt, convert
    if format == "alpaca":
        save_data = convert_to_alpaca(data)
    else:
        save_data = data

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    # Save metadata
    meta = {
        "name": name,
        "format": format,
        "record_count": len(data),
        "created_at": str(pd.Timestamp.now()),
    }
    with open(dataset_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return str(file_path)


def load_dataset(name: str) -> dict:
    """从本地加载数据集"""
    dataset_dir = DATASETS_DIR / name

    data_file = dataset_dir / "data.json"
    if not data_file.exists():
        return {"success": False, "data": [], "error": "数据集不存在"}

    try:
        with open(data_file, encoding="utf-8") as f:
            data = json.load(f)

        meta = {}
        meta_file = dataset_dir / "meta.json"
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)

        return {
            "success": True,
            "data": data,
            "format": meta.get("format", detect_format(data)),
            "record_count": len(data),
            "name": meta.get("name", name),
            "error": "",
        }
    except Exception as e:
        return {"success": False, "data": [], "error": str(e)}


def list_local_datasets() -> list[dict]:
    """列出所有本地已保存的数据集"""
    datasets = []
    if not DATASETS_DIR.exists():
        return datasets

    for ds_dir in DATASETS_DIR.iterdir():
        if not ds_dir.is_dir():
            continue
        meta_file = ds_dir / "meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                # Calculate file size
                data_file = ds_dir / "data.json"
                size_mb = 0
                if data_file.exists():
                    size_mb = round(data_file.stat().st_size / (1024 ** 2), 2)
                meta["size_mb"] = size_mb
                meta["id"] = ds_dir.name
                datasets.append(meta)
            except Exception:
                pass

    return datasets


def delete_local_dataset(name: str) -> dict:
    """删除本地数据集"""
    dataset_dir = DATASETS_DIR / name
    if not dataset_dir.exists():
        return {"success": False, "error": "数据集不存在"}

    try:
        shutil.rmtree(dataset_dir)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📚 内置数据集库")
    print("=" * 60)
    for ds in BUILTIN_DATASETS:
        print(f"  📊 {ds['name']}")
        print(f"     {ds['record_count']:,} 条 · {ds['size_mb']}MB · {ds['license']}")
        print(f"     {ds['description'][:60]}...")

    print()
    print("=" * 60)
    print("🔄 格式检测测试")
    print("=" * 60)

    sharegpt_sample = [{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}]
    alpaca_sample = [{"instruction": "写一首诗", "input": "", "output": "床前明月光..."}]
    conv_sample = [{"conversations": [{"from": "human", "value": "Hi"}, {"from": "assistant", "value": "Hello"}]}]

    print(f"  ShareGPT: {detect_format(sharegpt_sample)}")
    print(f"  Alpaca:   {detect_format(alpaca_sample)}")
    print(f"  Conv:     {detect_format(conv_sample)}")

    print()
    print("=" * 60)
    print("🔄 格式转换测试 (Alpaca → ShareGPT)")
    print("=" * 60)
    converted = convert_to_sharegpt(alpaca_sample)
    print(f"  {json.dumps(converted[0], ensure_ascii=False, indent=2)}")

    print()
    print("=" * 60)
    print("🧹 数据清洗测试")
    print("=" * 60)
    dirty_data = [
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},  # dup
        {"messages": []},  # empty
        {"messages": [{"role": "user", "content": "短"}, {"role": "assistant", "content": "太短"}]},  # too short
    ]
    result = clean_dataset(dirty_data)
    print(f"  清洗前: {result['stats']['total']} 条")
    print(f"  清洗后: {len(result['cleaned_data'])} 条")
    print(f"  移除: 重复{result['stats']['duplicate']} 空{result['stats']['empty']} 太短{result['stats']['too_short']}")

    print()
    print("=" * 60)
    print("📊 数据统计")
    print("=" * 60)
    stats = get_data_stats(sharegpt_sample)
    print(f"  {json.dumps(stats, ensure_ascii=False, indent=2)}")

    print()
    print("✅ 数据集管理模块自检完成")
