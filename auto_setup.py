#!/usr/bin/env python3
"""
模型训练室 · 自动环境初始化

首次运行时自动检测并安装所需工具：
- LLaMA-Factory（训练引擎）
- llama.cpp（GGUF 转换）
- Ollama（本地推理）

用法：python auto_setup.py
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
TOOLS_DIR = PROJECT_ROOT / "tools"
TOOLS_DIR.mkdir(exist_ok=True)

# ============================================================
# Helpers
# ============================================================

def green(s):  return f"\033[92m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"


def step(msg):
    print(f"\n{bold('▶')} {msg}")


def ok(msg):
    print(f"  {green('✅')} {msg}")


def warn(msg):
    print(f"  {yellow('⚠️')}  {msg}")


def fail(msg):
    print(f"  {red('❌')} {msg}")


def run(cmd, **kwargs):
    """运行命令，打印输出"""
    return subprocess.run(cmd, shell=True, **kwargs)


# ============================================================
# Checkers
# ============================================================

def check_python():
    step("检查 Python 环境")
    ver = sys.version_info
    if ver < (3, 10):
        fail(f"Python {ver.major}.{ver.minor} — 需要 ≥ 3.10")
        return False
    ok(f"Python {ver.major}.{ver.minor}.{ver.micro}")
    return True


def check_venv():
    step("检查虚拟环境")
    venv = PROJECT_ROOT / ".venv"
    if venv.exists() and (venv / "bin" / "python").exists():
        ok("虚拟环境已存在")
        return True
    warn("虚拟环境不存在，正在创建...")
    r = run(f"{sys.executable} -m venv .venv", cwd=PROJECT_ROOT)
    if r.returncode == 0:
        ok("虚拟环境已创建")
        return True
    fail("创建失败")
    return False


def check_pip_deps():
    step("检查 Python 依赖")
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = sys.executable

    try:
        import gradio
        ok("核心依赖已安装")
        return True
    except ImportError:
        warn("正在安装依赖...")
        r = run(
            f"{venv_python} -m pip install -r requirements.txt -q",
            cwd=PROJECT_ROOT, timeout=600,
        )
        if r.returncode == 0:
            ok("依赖安装完成")
            return True
        fail("安装失败")
        return False


def check_llama_factory():
    step("检查 LLaMA-Factory（训练引擎）")
    factory = PROJECT_ROOT / "LLaMA-Factory"
    if factory.exists() and (factory / "src" / "train.py").exists():
        ok(f"已安装: {factory}")
        return True

    warn("未安装。LLaMA-Factory 是训练功能的核心依赖。")
    ans = input("  是否自动安装？[Y/n] ").strip().lower()
    if ans == 'n':
        warn("跳过安装（训练功能将不可用）")
        return False

    print("  正在克隆 LLaMA-Factory...")
    r = run("git clone https://github.com/hiyouga/LLaMA-Factory.git --depth 1", cwd=PROJECT_ROOT)
    if r.returncode != 0:
        fail("克隆失败，请手动安装")
        return False

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = sys.executable
    print("  正在安装 LLaMA-Factory 依赖...")
    r = run(f"{venv_python} -m pip install -e LLaMA-Factory/ -q", cwd=PROJECT_ROOT, timeout=600)
    if r.returncode == 0:
        ok("LLaMA-Factory 安装完成")
        return True
    warn("依赖安装可能有警告，但已克隆完成")
    return True


def check_llama_cpp():
    step("检查 llama.cpp（GGUF 转换工具）")
    cpp = PROJECT_ROOT / "tools" / "llama.cpp"
    script = cpp / "convert_hf_to_gguf.py"
    if script.exists():
        ok(f"转换脚本已就绪: {script}")
        return True

    warn("未安装。GGUF 导出功能需要它。")
    ans = input("  是否自动下载转换脚本？[Y/n] ").strip().lower()
    if ans == 'n':
        warn("跳过（GGUF 导出仅支持 Python 量化）")
        return False

    print("  正在下载 llama.cpp...")
    r = run("git clone https://github.com/ggerganov/llama.cpp.git --depth 1 tools/llama.cpp", cwd=PROJECT_ROOT)
    if r.returncode == 0:
        ok("llama.cpp 下载完成")
        return True
    fail("下载失败")
    return False


def check_ollama():
    step("检查 Ollama（本地推理引擎）")
    from backend.env_config import get_ollama_bin, is_ollama_available
    if is_ollama_available():
        ok(f"Ollama 已可用: {get_ollama_bin()}")
        return True

    warn("未安装。AI 数据生成和本地推理需要它。")
    # Check if already downloaded
    extract = TOOLS_DIR / "ollama_extract" / "bin" / "ollama"
    if extract.exists():
        ok(f"Ollama 已下载（免安装版本）: {extract}")
        return True

    ans = input("  是否自动下载 Ollama（~2GB）？[Y/n] ").strip().lower()
    if ans == 'n':
        warn("跳过（AI 数据生成、本地推理不可用）")
        return False

    import platform
    system = platform.system().lower()
    arch = platform.machine().lower()
    if arch == "x86_64":
        arch = "amd64"

    print(f"  系统: {system}/{arch}")
    tgz_path = TOOLS_DIR / "ollama.tgz"

    try:
        url = f"https://github.com/ollama/ollama/releases/latest/download/ollama-{system}-{arch}.tgz"
        print(f"  正在下载 {url}...")
        urllib.request.urlretrieve(url, tgz_path)
    except Exception as e:
        fail(f"下载失败: {e}")
        print(f"  手动下载: https://ollama.com/download")
        return False

    # Extract
    print("  正在解压...")
    extract_dir = TOOLS_DIR / "ollama_extract"
    extract_dir.mkdir(exist_ok=True)
    import tarfile
    with tarfile.open(tgz_path) as tf:
        tf.extractall(extract_dir)
    tgz_path.unlink()  # Clean up archive

    if (extract_dir / "bin" / "ollama").exists():
        ok("Ollama 安装完成")
        return True
    fail("解压失败")
    return False


# ============================================================
# Data directories
# ============================================================

def check_dirs():
    step("创建数据目录")
    dirs = [
        "data/models",
        "data/datasets",
        "data/experiments",
        "data/exports",
    ]
    for d in dirs:
        p = PROJECT_ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        ok(d)
    return True


def check_sitecustomize():
    """Python 3.14 pickle 兼容性修复"""
    step("检查 Python 3.14 兼容性补丁")
    if sys.version_info < (3, 14):
        ok("Python < 3.14，不需要补丁")
        return True

    import site
    site_dir = Path(site.getsitepackages()[0])
    sc = site_dir / "sitecustomize.py"
    source = PROJECT_ROOT / "tests" / "fix_py314_pickle.py"

    if sc.exists():
        ok("sitecustomize.py 已安装")
        return True

    warn("Python 3.14 需要 pickle 兼容性补丁")
    try:
        import shutil
        shutil.copy(source, sc)
        ok(f"补丁已安装到 {sc}")
        return True
    except PermissionError:
        fail(f"无权限写入 {site_dir}")
        print(f"  请手动复制: cp {source} {sc}")
        return False


# ============================================================
# Main
# ============================================================

def main():
    print()
    print(bold("=" * 55))
    print(bold("  模型训练室 · 自动环境初始化"))
    print(bold("=" * 55))

    checks = [
        ("Python 环境", check_python),
        ("虚拟环境", check_venv),
        ("Python 依赖", check_pip_deps),
        ("数据目录", check_dirs),
        ("Python 3.14 补丁", check_sitecustomize),
        ("LLaMA-Factory（训练引擎）", check_llama_factory),
        ("llama.cpp（GGUF 导出）", check_llama_cpp),
        ("Ollama（本地推理）", check_ollama),
    ]

    results = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as e:
            fail(f"检测异常: {e}")
            results[name] = False

    print()
    print(bold("=" * 55))
    print(bold("  初始化结果"))
    print(bold("=" * 55))

    all_ok = True
    core_ok = True
    for name, ok_flag in results.items():
        icon = green("✅") if ok_flag else yellow("⚠️")
        if "训练引擎" in name and not ok_flag:
            core_ok = False
        print(f"  {icon} {name}")

    print()
    if all_ok:
        print(green("🎉 所有组件就绪！运行 ./run.sh 启动"))
    elif core_ok:
        print(yellow("⚠️  核心功能可用，部分可选组件缺失"))
        print("  运行 ./run.sh 启动")
    else:
        print(red("❌ 核心组件缺失，请先解决上述问题"))

    print(f"\n  💡 设置环境变量可覆盖默认路径：")
    print(f"     MTR_OLLAMA_BIN        Ollama 可执行文件路径")
    print(f"     MTR_LLAMA_FACTORY_PATH LLaMA-Factory 路径")
    print(f"     MTR_MODELS_DIR         模型存储目录")
    print(f"     MTR_DATASETS_DIR       数据集存储目录")


if __name__ == "__main__":
    main()
