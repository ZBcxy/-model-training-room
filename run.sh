#!/usr/bin/env bash
# ==============================================================
# 模型训练室 · Model Training Room — 一键启动脚本
# ==============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  🏠  模型训练室 · Model Training Room                       ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}     下载 · 准备数据 · 微调 · 导出                          ${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ----------------------------------------------------------
# 1. Check Python
# ----------------------------------------------------------
echo -e "${YELLOW}[1/5]${NC} 检查 Python 环境..."

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ 未找到 python3。请安装 Python 3.10+${NC}"
    exit 1
fi

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "  ${GREEN}✅${NC} Python $PYTHON_VER"

# ----------------------------------------------------------
# 2. Virtual Environment
# ----------------------------------------------------------
echo -e "${YELLOW}[2/5]${NC} 检查虚拟环境..."

if [ ! -d ".venv" ]; then
    echo "  创建虚拟环境..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo -e "  ${GREEN}✅${NC} 虚拟环境就绪"

# ----------------------------------------------------------
# 3. Dependencies
# ----------------------------------------------------------
echo -e "${YELLOW}[3/5]${NC} 检查依赖..."

# Quick check if key packages are installed
if ! python -c "import gradio" 2>/dev/null; then
    echo "  安装 Python 依赖（这可能需要几分钟）..."
    pip install -r requirements.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
fi

echo -e "  ${GREEN}✅${NC} 依赖就绪"

# ----------------------------------------------------------
# 4. Check LLaMA-Factory (optional but recommended)
# ----------------------------------------------------------
echo -e "${YELLOW}[4/5]${NC} 检查训练引擎 (LLaMA-Factory)..."

if [ ! -d "LLaMA-Factory" ]; then
    echo ""
    echo -e "  ${YELLOW}⚠️${NC}  LLaMA-Factory 未安装（训练功能需要它）"
    echo "  是否安装？[Y/n] "
    read -r answer
    if [ "$answer" != "n" ] && [ "$answer" != "N" ]; then
        echo "  克隆 LLaMA-Factory..."
        git clone https://github.com/hiyouga/LLaMA-Factory.git --depth 1
        echo "  安装 LLaMA-Factory 依赖..."
        pip install -e LLaMA-Factory/ -q -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn 2>/dev/null
        echo -e "  ${GREEN}✅${NC} LLaMA-Factory 安装完成"
    else
        echo -e "  ${YELLOW}⚠️${NC}  跳过安装。训练功能将不可用。"
    fi
else
    echo -e "  ${GREEN}✅${NC} LLaMA-Factory 已安装"
fi

# ----------------------------------------------------------
# 5. GPU Check & Launch
# ----------------------------------------------------------
echo -e "${YELLOW}[5/5]${NC} 检测硬件..."

GPU_INFO=$(python -c "
from backend.hardware_checker import get_system_info
info = get_system_info()
if info.has_gpu:
    gpu = info.gpus[0]
    print(f'GPU: {gpu.name} ({gpu.vram_total_gb:.1f}GB)')
else:
    print('GPU: 未检测到')
print(f'RAM: {info.ram_total_gb:.1f}GB')
print(f'Disk: {info.disk_free_gb:.1f}GB free')
" 2>/dev/null || echo "GPU: 检测失败")

echo -e "  ${GREEN}$GPU_INFO${NC}"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  🚀 启动模型训练室...                                     ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  浏览器打开: http://127.0.0.1:7860                       ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  按 Ctrl+C 停止                                          ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

python -m frontend.app
