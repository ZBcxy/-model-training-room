# ============================================================
# 模型训练室 Model Training Room — Docker 镜像
#
# 基于 PyTorch 官方 CUDA 镜像，预装所有训练依赖
#
# 构建:
#   docker build -t model-training-room .
#
# 运行:
#   docker run --gpus all -p 7860:7860 -v $(pwd)/data:/app/data model-training-room
# ============================================================

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL maintainer="ZBcxy"
LABEL description="模型训练室 — 本地化大模型微调一站式工具"
LABEL version="0.2.0"

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# ----------------------------------------------------------
# 系统依赖
# ----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# Python 依赖（分层缓存）
# ----------------------------------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir bitsandbytes

# ----------------------------------------------------------
# LLaMA-Factory（训练引擎）
# ----------------------------------------------------------
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git --depth 1 /opt/llama-factory && \
    cd /opt/llama-factory && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || true

# ----------------------------------------------------------
# 项目源码
# ----------------------------------------------------------
COPY . .

# 创建数据目录
RUN mkdir -p data/models data/datasets data/experiments data/exports

# ----------------------------------------------------------
# 默认环境变量
# ----------------------------------------------------------
ENV MTR_LLAMA_FACTORY_PATH=/opt/llama-factory
ENV MTR_MODELS_DIR=/app/data/models
ENV MTR_DATASETS_DIR=/app/data/datasets
ENV MTR_EXPERIMENTS_DIR=/app/data/experiments
ENV MTR_EXPORTS_DIR=/app/data/exports

# Ollama 需要单独运行，不在容器内
ENV MTR_OLLAMA_HOST=http://host.docker.internal:11434

# ----------------------------------------------------------
# 暴露端口 + 启动
# ----------------------------------------------------------
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

CMD ["python", "-m", "frontend.app"]
