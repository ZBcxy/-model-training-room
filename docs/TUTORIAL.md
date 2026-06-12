# 模型训练室 · 使用教程

5 步完成模型微调。不需要写代码。

## 环境要求

- NVIDIA GPU（6GB 显存起步，12GB 推荐）
- Python 3.10+
- 50GB 磁盘空间

## 快速开始

```bash
git clone https://github.com/ZBcxy/-model-training-room.git
cd model-training-room
python auto_setup.py   # 自动检测+安装依赖
python -m frontend.app  # 启动
# 浏览器打开 http://127.0.0.1:7860
```

## Step 1: 选择模型

1. 打开「模型」页面
2. 搜索你想要的基础模型（如 Qwen2.5-7B）
3. 点击下载

> 推荐入门模型：Qwen2.5-1.5B-Instruct（3GB，下载快，显存友好）

## Step 2: 准备数据

三种方式任选其一：

**方式 A：内置数据集**
- 数据页下拉选择，一键使用

**方式 B：上传文件**
- 支持 JSON/JSONL/CSV
- 自动检测格式并转换

**方式 C：AI 生成**
- 写几个问答示例
- 点「生成」，本地 Ollama 自动扩充到 100+ 条

## Step 3: 配置训练

1. 输入模型 ID，点「匹配」获取推荐参数
2. 选择微调方式：LoRA（推荐）或 QLoRA（显存紧张时）
3. 点「生成配置」查看预览
4. 点「开始训练」

## Step 4: 监控训练

- 切换到「结果」页面
- 输入实验 ID
- 查看实时 Loss 曲线

## Step 5: 导出部署

1. 结果页 → 输入模型路径 → 选择量化级别 → 导出 GGUF
2. 用 Ollama 加载：
```bash
ollama create my-model -f Modelfile
ollama run my-model
```

## 常见问题

**Q: 训练 OOM（显存不足）？**
A: ① 改用 QLoRA ② 减小 batch_size ③ 减小 max_seq_length

**Q: 下载模型很慢？**
A: 尝试在搜索时勾选 ModelScope（国内更快）

**Q: 训练效果不好？**
A: ① 增加数据量（至少 500 条）② 检查数据质量 ③ 调整学习率
