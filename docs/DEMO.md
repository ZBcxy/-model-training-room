# 模型训练室 · 5分钟快速演示

> 从零开始，5分钟完成一个中文对话模型的微调。

## 环境检查

```bash
$ git clone https://github.com/ZBcxy/-model-training-room.git
$ cd model-training-room
$ python auto_setup.py

  ▶ 检查 Python 环境
    ✅ Python 3.11.5
  ▶ 检查虚拟环境
    ✅ 虚拟环境已存在
  ▶ 检查 Python 依赖
    ✅ 核心依赖已安装
  ▶ 检查 LLaMA-Factory（训练引擎）
    ✅ 已安装
  ▶ 检查 Ollama（本地推理）
    ✅ Ollama 已可用
  ▶ 检查 llama.cpp（GGUF 导出）
    ✅ 转换脚本已就绪

  🎉 所有组件就绪！
```

## Step 1: 下载模型（1分钟）

```bash
$ python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='data/models/Qwen--Qwen2.5-1.5B-Instruct')
"

  📥 下载中... Qwen2.5-1.5B-Instruct (2.88 GB)
  ✅ 完成: data/models/Qwen--Qwen2.5-1.5B-Instruct/
```

## Step 2: 准备数据（30秒）

```bash
$ python tests/gen_big_dataset.py

  ✅ 生成 500 条数据
     平均长度: 121 字符
     保存: data/datasets/demo-2k-zh/data.json
```

## Step 3: 训练（2分钟）

```bash
$ python -c "
from backend.training_engine import *
config = create_training_config(
    'Qwen/Qwen2.5-1.5B-Instruct',
    'data/models/Qwen--Qwen2.5-1.5B-Instruct',
    'data/datasets/demo-2k-zh/data.json',
    finetuning_type='lora', preset='standard',
    available_vram_gb=11.6, dataset_size=500,
)
from backend.training_engine import TrainingExecutor
executor = TrainingExecutor(config)
executor.start()
"

  🚀 训练启动
  ✅ experiment_id: exp_1718...
  📊 Loss: 2.15 → 0.39
  ⏱ 耗时: 82 秒
  💾 Checkpoint: data/experiments/compare-qwen1.5b/
```

## Step 4: 导出 GGUF（30秒）

```bash
$ python -m llamafactory.cli export \
    --model_name_or_path data/models/Qwen--Qwen2.5-1.5B-Instruct \
    --adapter_name_or_path data/experiments/compare-qwen1.5b \
    --template qwen --finetuning_type lora \
    --export_dir data/exports/merged

  🔀 合并 LoRA 权重... ✅
  📦 导出完成

$ python tools/llama.cpp/convert_hf_to_gguf.py \
    data/exports/merged --outfile model-f16.gguf --outtype f16

  ✅ Model exported to model-f16.gguf (2.9 GB)

$ python tools/quantize_gguf.py model-f16.gguf model-q4km.gguf --type q4_K_M

  ✅ 量化完成: 2.9 GB → 0.92 GB (32%)
```

## Step 5: 部署到 Ollama（10秒）

```bash
$ cat > Modelfile << 'EOF'
FROM ./model-q4km.gguf
TEMPLATE """<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
SYSTEM """你是一个乐于助人的中文AI助手。"""
EOF

$ ollama create my-assistant --file Modelfile
  ✅ success

$ ollama run my-assistant "中国四大发明是什么？"

  指南针、火药、造纸术和印刷术。这些发明对世界文明
  进程产生了深远影响。指南针推动了航海探险，火药改变
  了战争形态，造纸术和印刷术让知识得以广泛传播...
```

## 效果展示

```
模型: Qwen2.5-1.5B-Instruct + LoRA 微调
数据: 500 条中文对话
训练: 82 秒
大小: 0.92 GB (Q4_K_M)
显存: <4 GB

测试 1 — 知识问答:
  Q: 中国四大发明是什么？
  A: 指南针、火药、造纸术和印刷术。这些发明对世界文明
     进程产生了深远影响... ✅

测试 2 — 日常对话:
  Q: 今天好累，有什么建议？
  A: 可以试试深呼吸、散步、看喜欢的书，照顾好自己最重要！ ✅

测试 3 — 技术解释:
  Q: 什么是机器学习？
  A: 机器学习是让计算机从数据中自动学习规律的方法...
     ✅
```

## 核心数据

| 指标 | 值 |
|------|-----|
| 基础模型 | Qwen2.5-1.5B-Instruct |
| 微调方式 | LoRA (rank=16) |
| 训练数据 | 500 条 |
| 训练时间 | 82 秒 |
| 最终 Loss | 0.39 |
| 显存占用 | <4 GB |
| 导出大小 | 0.92 GB (Q4_K_M) |
| 推理速度 | ~15 tokens/s (RTX 3060) |

## 就是这么快

从下载模型到部署可用，总共 **不到5分钟**。

[完整训练报告 →](TRAINING_REPORT.md) · [GPU环境指南 →](GPU_SETUP.md) · [贡献指南 →](../CONTRIBUTING.md)
