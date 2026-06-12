# 🏠 模型训练室 · Model Training Room

> 下载 · 准备数据 · 微调 · 导出 —— 一站式模型微调工具
>
> Download · Prepare Data · Fine-tune · Export — All-in-one model fine-tuning tool

---

[English](#english) | [中文](#中文)

---

# English

## What is Model Training Room?

Model Training Room is a **local-first, all-in-one model fine-tuning tool** that transforms the complex workflow of fine-tuning large language models from "read docs + write scripts + debug YAML + stare at tracebacks" into **"pick → configure → click → wait → use"**.

It provides a graphical interface for:
- 🔍 Searching and downloading open-source/ commercially-licensed models from Hugging Face, ModelScope, and more
- 📚 Preparing training data via built-in datasets, online search, or AI-powered data generation
- 🧪 Fine-tuning models using LoRA / QLoRA / Full fine-tuning with smart parameter recommendations
- 🔬 Real-time training monitoring with loss curves and GPU status
- 📦 Exporting models to GGUF format for Ollama / llama.cpp deployment

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | Python FastAPI | The native language of ML, no alternatives |
| Frontend (MVP) | Gradio | Fastest prototyping framework for ML apps |
| Model Download | huggingface_hub + modelscope SDK | Covers both global and Chinese model hubs |
| Fine-tuning Engine | LLaMA-Factory (Python API) | Most mature open-source fine-tuning framework, supports 100+ model architectures |
| Dataset Search | Hugging Face Datasets + ModelScope | Access to 150K+ open datasets |
| Export | llama.cpp (GGUF conversion + quantization) | Standard format for local inference |
| Database | SQLite | Zero-config, stores experiments, models, and training history |

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd "Model Training Room"

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install LLaMA-Factory for training
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e . && cd ..

# 5. Launch the app
python -m frontend.app
# Open http://127.0.0.1:7860 in your browser
```

## Project Structure

```
Model-Training-Room/
├── CLAUDE.md                  # Project overview document (for AI agents)
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── backend/
│   ├── __init__.py
│   ├── hardware_checker.py    # GPU/CPU detection + VRAM budget calculator
│   ├── model_hub.py           # Multi-source model search & download
│   ├── dataset_manager.py     # Dataset import, cleaning, generation wizard
│   ├── training_engine.py     # LLaMA-Factory wrapper + smart recommendations
│   ├── training_monitor.py    # Real-time training monitoring
│   ├── export.py              # GGUF export + Model Card generation
│   └── experiment_store.py    # SQLite-based experiment management
├── frontend/
│   └── app.py                 # Gradio web UI (6 pages)
├── data/
│   ├── models/                # Downloaded model cache
│   ├── datasets/              # Training datasets
│   └── experiments/           # Experiment records & checkpoints
└── .venv/                     # Virtual environment
```

## Features

### 0️⃣ Environment Check
- Auto-detect GPU model / VRAM / CUDA version / RAM / disk
- Environment dependency check (PyTorch, CUDA toolkit)
- VRAM budget calculator — know what you can run before you start

### 1️⃣ Model Hub
- **Multi-source search**: Hugging Face + ModelScope unified search
- **License auto-tagging**: 🟢 Commercial / 🟡 Review needed / 🔴 Non-commercial
- **Resumable downloads**: Never re-download from scratch
- **Local model library**: Manage downloaded models, check disk usage

### 2️⃣ Data Preparation (Core Competency)
Three paths to get training data:
- **Path 1 — Built-in Library**: 10+ curated, ready-to-use datasets with previews
- **Path 2 — Online Search**: Search HuggingFace + ModelScope datasets directly
- **Path 3 — AI Generation Wizard**: Write 3-5 examples → AI generates 1000+ more

Data tools:
- Auto format detection & conversion (Alpaca ↔ ShareGPT ↔ Conversation)
- One-click cleaning (dedup, remove empty, truncation)
- Quality report (length distribution, language detection)
- Train/validation split

### 3️⃣ Training Configuration
- **3-step wizard**: Pick Model → Select Data → Tune Parameters → Go
- **Smart recommendations**: Auto-suggest LoRA rank, learning rate, batch size based on VRAM
- **Presets**: Quick (1h) / Standard (3h) / Deep (8h) / Custom
- **Chat Template auto-detection**: ChatML, Llama3, Mistral, Gemma, etc.
- **VRAM budget visualization**: See exactly where your GPU memory goes

### 4️⃣ Training Monitor
- Real-time loss curve + learning rate chart
- GPU status (utilization, memory, temperature)
- Live sample outputs every N steps
- Checkpoint management
- Pause / Resume / Stop with auto-save

### 5️⃣ Evaluation & Export
- **Chat playground**: Test your fine-tuned model directly in the UI
- **GGUF export**: with quantization options (Q4_0, Q4_K_M, Q5_K_M, Q8_0, FP16)
- **Model Card auto-generation**: HuggingFace-compatible model card
- **Ollama Modelfile generation**: Ready-to-use Ollama deployment config

### 6️⃣ Experiment Management
- Complete training history with all parameters
- Compare experiments side-by-side
- Delete / Re-run with same config

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM (QLoRA) | 6 GB | 12+ GB |
| GPU VRAM (LoRA) | 16 GB | 24+ GB |
| GPU VRAM (Full) | 40 GB | 80+ GB |
| RAM | 16 GB | 32+ GB |
| Disk | 50 GB free | 200+ GB free |
| Python | 3.10+ | 3.11+ |
| CUDA | 11.8+ | 12.1+ |

## Roadmap

### MVP (Current — v0.1)
- [x] Hardware detection & VRAM budget
- [x] HF + ModelScope model search & download
- [x] Dataset import, auto-format, preview
- [x] Smart training parameter recommendations
- [x] Real-time training monitoring
- [x] Chat testing + GGUF export
- [x] Basic experiment management

### v0.2
- [ ] AI data generation wizard (Paths B & C)
- [ ] Expanded built-in dataset library
- [ ] Multi-GPU support (DeepSpeed/FSDP)
- [ ] Task queue for batch training
- [ ] A/B comparison
- [ ] Offline mode
- [ ] Model Card auto-upload

### Future
- [ ] Data quality reports with visualizations
- [ ] ONNX export
- [ ] Auto early-stopping
- [ ] Multi-user web collaboration
- [ ] Docker one-click deployment
- [ ] React + FastAPI production frontend

## Design Decisions

### Why Gradio instead of Electron?
Gradio enables fastest prototyping for ML apps. Can upgrade to React + FastAPI later.

### Why LLaMA-Factory instead of raw Transformers?
LLaMA-Factory has already solved 100+ model compatibility issues, training templates, and data format adaptations. Building from scratch would duplicate this work.

### Why is the Data Generation Wizard the core competency?
Most fine-tuning tools only solve "how to train", not "what to train with". Data preparation is the most time-consuming and frustrating part of fine-tuning. Providing zero-barrier data acquisition is the key differentiator.

### Why license tagging matters?
Open-source model licenses are complex (Apache 2.0, MIT, CC, Llama License, etc.). Most users can't determine commercial usability. Auto-tagging + risk warnings prevent legal issues.

---

---

# 中文

## 模型训练室是什么？

模型训练室是一款**本地化模型微调一站式工具**，将复杂的大模型微调工作流从「读文档 + 写脚本 + 调 YAML + 看 traceback」变成**「挑选 + 配置 + 点击 + 等待 + 使用」**。

提供图形化界面完成：
- 🔍 从 Hugging Face、ModelScope 等平台搜索和下载开源/可商用模型
- 📚 通过内置数据集、在线搜索或 AI 生成向导准备训练数据
- 🧪 使用 LoRA / QLoRA / 全参微调进行模型训练，智能参数推荐
- 🔬 实时训练监控，Loss 曲线和 GPU 状态可视化
- 📦 导出 GGUF 格式模型，部署到 Ollama / llama.cpp

## 技术栈

| 层面 | 选择 | 理由 |
|------|------|------|
| 后端 | Python FastAPI | ML 生态原生语言，无需解释 |
| 前端（MVP） | Gradio | ML 项目最快速的原型框架 |
| 模型下载 | huggingface_hub + modelscope SDK | 覆盖国内外两大模型源 |
| 微调引擎 | LLaMA-Factory（Python API） | 社区最成熟，支持 100+ 模型架构 |
| 数据集搜索 | Hugging Face Datasets + ModelScope | 15 万+ 开放数据集 |
| 导出 | llama.cpp（GGUF 转换 + 量化） | 本地推理的标准格式 |
| 数据库 | SQLite | 零配置，存储实验、模型和训练历史 |

## 快速开始

```bash
# 1. 克隆仓库
git clone <repo-url>
cd "Model Training Room"

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4.（可选）安装 LLaMA-Factory 用于训练
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e . && cd ..

# 5. 启动应用
python -m frontend.app
# 浏览器打开 http://127.0.0.1:7860
```

## 项目结构

```
Model-Training-Room/
├── CLAUDE.md                  # 项目全貌文档（给 AI 智能体看的）
├── README.md                  # 本文件
├── requirements.txt           # Python 依赖
├── backend/
│   ├── __init__.py
│   ├── hardware_checker.py    # GPU/CPU 检测 + 显存预算计算器
│   ├── model_hub.py           # 多源模型搜索与下载
│   ├── dataset_manager.py     # 数据集导入、清洗、生成向导
│   ├── training_engine.py     # LLaMA-Factory 封装 + 智能推荐
│   ├── training_monitor.py    # 实时训练监控
│   ├── export.py              # GGUF 导出 + Model Card 生成
│   └── experiment_store.py    # SQLite 实验管理
├── frontend/
│   └── app.py                 # Gradio Web 界面（6 页）
├── data/
│   ├── models/                # 下载的模型缓存
│   ├── datasets/              # 训练数据集
│   └── experiments/           # 实验记录与 checkpoint
└── .venv/                     # 虚拟环境
```

## 功能详解

### 0️⃣ 环境检测
- 自动检测 GPU 型号 / 显存 / CUDA 版本 / 内存 / 磁盘
- 环境依赖检查（PyTorch、CUDA toolkit 等）
- 显存预算计算器 — 动手前就知道能跑多大的模型

### 1️⃣ 模型中心
- **多源搜索**：Hugging Face + ModelScope 统一搜索
- **许可证自动标注**：🟢 可商用 / 🟡 需审查 / 🔴 不可商用
- **断点续传**：大模型下载不怕断网
- **本地模型库**：管理已下载模型，查看磁盘占用

### 2️⃣ 数据准备（核心竞争力）
三条数据获取路径：
- **路径一 — 内置数据集库**：10+ 精选数据集，带预览，一键使用
- **路径二 — 在线搜索**：直接在 HuggingFace / ModelScope 搜索数据集
- **路径三 — AI 生成向导**：写 3-5 个示例 → AI 批量扩充到 1000+ 条

数据处理工具：
- 格式自动检测与转换（Alpaca ↔ ShareGPT ↔ Conversation）
- 一键清洗（去重、去空、截断）
- 质量报告（长度分布、语言检测）
- 训练/验证集自动切分

### 3️⃣ 训练配置
- **三步向导**：选模型 → 配数据 → 调参数 → 开始训练
- **智能推荐**：根据显存自动推荐 LoRA rank、学习率、batch size
- **预设方案**：快速尝试 (1h) / 标准微调 (3h) / 深度训练 (8h) / 自定义
- **Chat Template 自动检测**：ChatML / Llama3 / Mistral / Gemma 等
- **显存预算可视化**：清楚看到 GPU 显存都花在哪里

### 4️⃣ 训练监控
- 实时 Loss 曲线 + 学习率变化
- GPU 状态（利用率、显存、温度）
- 每 N 步采样输出预览
- Checkpoint 管理
- 暂停 / 恢复 / 停止（自动保存）

### 5️⃣ 评估与导出
- **聊天测试**：直接在界面里和微调后的模型对话
- **GGUF 导出**：支持多种量化级别（Q4_0 / Q4_K_M / Q5_K_M / Q8_0 / FP16）
- **Model Card 自动生成**：兼容 HuggingFace 标准的模型卡片
- **Ollama Modelfile 生成**：一键部署到 Ollama

### 6️⃣ 实验管理
- 完整训练历史，所有参数可追溯
- 实验对比
- 删除 / 用相同配置重新训练

## 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU 显存 (QLoRA) | 6 GB | 12+ GB |
| GPU 显存 (LoRA) | 16 GB | 24+ GB |
| GPU 显存 (全参) | 40 GB | 80+ GB |
| 内存 | 16 GB | 32+ GB |
| 磁盘 | 50 GB 可用 | 200+ GB 可用 |
| Python | 3.10+ | 3.11+ |
| CUDA | 11.8+ | 12.1+ |

## 开发路线图

### MVP（当前 — v0.1）
- [x] 硬件检测与显存预算
- [x] HF + ModelScope 模型搜索与下载
- [x] 数据集导入、自动格式转换、预览
- [x] 智能训练参数推荐
- [x] 实时训练监控
- [x] 对话测试 + GGUF 导出
- [x] 基础实验管理

### v0.2
- [ ] AI 数据生成向导（路径 B 和 C）
- [ ] 内置数据集库扩充
- [ ] 多 GPU 支持（DeepSpeed/FSDP）
- [ ] 批量训练任务队列
- [ ] A/B 对比完善
- [ ] 离线模式
- [ ] Model Card 自动上传

### 远期
- [ ] 数据质量可视化报告
- [ ] ONNX 导出
- [ ] 自动早停机制
- [ ] 多用户 Web 协作
- [ ] Docker 一键部署
- [ ] React + FastAPI 生产级前端

## 设计决策

### 为什么用 Gradio 而不是 Electron？
MVP 阶段用 Gradio 开发速度最快，后续可升级为 React + FastAPI 的正式 Web 应用。

### 为什么用 LLaMA-Factory 而不是自己封装 Transformers？
LLaMA-Factory 已经解决了 100+ 模型兼容性、训练模板、数据格式适配等问题，自己封装工作量大且易出错。

### 为什么数据生成向导是核心竞争力？
大多数微调工具只解决「怎么训」，不解决「训什么」。数据准备是微调最费时最劝退的环节，提供零门槛的数据获取方案是差异化优势。

### 为什么许可证标注这么重要？
开源模型协议复杂（Apache 2.0、MIT、CC、Llama License 等），普通用户难以判断是否可以商用。自动标注 + 风险提示避免法律纠纷。

---

<p align="center">
  <b>Built with ❤️ for the open-source AI community</b><br>
  <b>为开源 AI 社区倾心打造</b>
</p>
