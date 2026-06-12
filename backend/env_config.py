"""
环境配置中心 — 解决硬编码路径问题

所有工具路径从这里统一管理，优先级：
1. 环境变量（MTR_*）
2. 自动检测
3. 默认相对路径
"""

import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def _detect(name: str, candidates: list) -> Path | None:
    """从候选路径中检测第一个存在的"""
    for c in candidates:
        if c is not None and c.exists():
            return c.resolve()
    return None


def _which(cmd: str) -> Path | None:
    """查找命令在 PATH 中的位置"""
    found = shutil.which(cmd)
    return Path(found).resolve() if found else None


# ============================================================
# Ollama
# ============================================================

def get_ollama_bin() -> Path | None:
    """获取 Ollama 可执行文件路径"""
    env = os.environ.get("MTR_OLLAMA_BIN", "")
    if env:
        p = Path(env)
        if p.exists():
            return p

    return _detect("ollama", [
        _which("ollama"),
        PROJECT_ROOT / "tools" / "ollama_extract" / "bin" / "ollama",
        Path.home() / ".ollama" / "bin" / "ollama",
        Path("/usr/local/bin/ollama"),
    ])


def get_ollama_host() -> str:
    """获取 Ollama 服务地址"""
    return os.environ.get("MTR_OLLAMA_HOST", "http://127.0.0.1:11434")


def is_ollama_available() -> bool:
    """Ollama 是否可用"""
    bin_path = get_ollama_bin()
    if not bin_path:
        return False
    try:
        result = subprocess.run([str(bin_path), "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


# ============================================================
# LLaMA-Factory
# ============================================================

def get_llama_factory_path() -> Path | None:
    """获取 LLaMA-Factory 安装路径"""
    env = os.environ.get("MTR_LLAMA_FACTORY_PATH", "")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # Try import
    try:
        import llamafactory
        return Path(llamafactory.__file__).parent.parent
    except ImportError:
        pass

    # Check local clone
    return _detect("LLaMA-Factory", [
        PROJECT_ROOT / "LLaMA-Factory",
        Path.home() / "LLaMA-Factory",
    ])


def is_llama_factory_available() -> bool:
    """LLaMA-Factory 是否可用"""
    try:
        import llamafactory
        return True
    except ImportError:
        pass
    path = get_llama_factory_path()
    if path and (path / "src" / "train.py").exists():
        return True
    return False


# ============================================================
# llama.cpp / GGUF 工具
# ============================================================

def get_convert_script() -> Path | None:
    """获取 convert_hf_to_gguf.py 路径"""
    env = os.environ.get("MTR_LLAMA_CPP_CONVERT", "")
    if env:
        p = Path(env)
        if p.exists():
            return p

    return _detect("convert_hf_to_gguf.py", [
        PROJECT_ROOT / "tools" / "llama.cpp" / "convert_hf_to_gguf.py",
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
    ])


def get_quantize_bin() -> Path | None:
    """获取 quantize 可执行文件路径"""
    env = os.environ.get("MTR_LLAMA_CPP_QUANTIZE", "")
    if env:
        p = Path(env)
        if p.exists():
            return p

    return _detect("quantize", [
        _which("llama-quantize"),
        _which("quantize"),
        PROJECT_ROOT / "tools" / "llama.cpp" / "build" / "bin" / "llama-quantize",
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
    ])


def is_gguf_export_available() -> bool:
    """GGUF 导出功能是否可用"""
    # Python-level quantization via llama-cpp-python always works
    try:
        import llama_cpp.llama_cpp
        return True
    except ImportError:
        return False


# ============================================================
# Models & Data
# ============================================================

def get_models_dir() -> Path:
    """获取模型存储目录"""
    env = os.environ.get("MTR_MODELS_DIR", "")
    if env:
        return Path(env)
    return PROJECT_ROOT / "data" / "models"


def get_datasets_dir() -> Path:
    """获取数据集存储目录"""
    env = os.environ.get("MTR_DATASETS_DIR", "")
    if env:
        return Path(env)
    return PROJECT_ROOT / "data" / "datasets"


def get_experiments_dir() -> Path:
    """获取实验存储目录"""
    env = os.environ.get("MTR_EXPERIMENTS_DIR", "")
    if env:
        return Path(env)
    return PROJECT_ROOT / "data" / "experiments"


def get_exports_dir() -> Path:
    """获取导出文件目录"""
    env = os.environ.get("MTR_EXPORTS_DIR", "")
    if env:
        return Path(env)
    return PROJECT_ROOT / "data" / "exports"


# ============================================================
# System Report
# ============================================================

def get_environment_report() -> dict:
    """生成完整的环境配置报告"""
    return {
        "project_root": str(PROJECT_ROOT),
        "ollama": {
            "available": is_ollama_available(),
            "binary": str(get_ollama_bin()) if get_ollama_bin() else None,
            "host": get_ollama_host(),
        },
        "llama_factory": {
            "available": is_llama_factory_available(),
            "path": str(get_llama_factory_path()) if get_llama_factory_path() else None,
        },
        "gguf_export": {
            "available": is_gguf_export_available(),
            "convert_script": str(get_convert_script()) if get_convert_script() else None,
            "quantize_bin": str(get_quantize_bin()) if get_quantize_bin() else None,
        },
        "directories": {
            "models": str(get_models_dir()),
            "datasets": str(get_datasets_dir()),
            "experiments": str(get_experiments_dir()),
            "exports": str(get_exports_dir()),
        },
    }


def print_environment_report():
    """打印人类可读的环境配置"""
    r = get_environment_report()
    print("=" * 60)
    print("模型训练室 · 环境配置")
    print("=" * 60)
    print(f"\n📁 项目根目录: {r['project_root']}")
    print(f"\n🦙 Ollama: {'✅' if r['ollama']['available'] else '❌'} {r['ollama']['binary'] or '未安装'}")
    print(f"🏭 LLaMA-Factory: {'✅' if r['llama_factory']['available'] else '❌'} {r['llama_factory']['path'] or '未安装'}")
    print(f"📦 GGUF导出: {'✅' if r['gguf_export']['available'] else '❌'}")
    print(f"\n📂 数据目录:")
    for k, v in r['directories'].items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    print_environment_report()
