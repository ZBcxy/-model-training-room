# GPU 环境配置完全指南

从零开始，把显卡配置成大模型微调工作站。

## 第一步：确认你的显卡

```bash
# Linux
nvidia-smi

# Windows
# 打开任务管理器 → 性能 → GPU
```

**关键指标：显存（VRAM）**。决定了你能微调多大的模型：

| 显存 | 能做什么 |
|------|----------|
| 4-6 GB | QLoRA 微调 ≤3B 模型 |
| 8-12 GB | QLoRA 微调 7B / LoRA 微调 ≤3B |
| 16-24 GB | LoRA 微调 7-13B / QLoRA 微调 30B+ |
| 40+ GB | 全参微调 7B / LoRA 微调 70B |

## 第二步：安装 CUDA Toolkit

### Ubuntu/Debian
```bash
# 推荐 CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --toolkit --silent
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Arch Linux
```bash
sudo pacman -S cuda cudnn
```

### Windows
1. 下载 [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)
2. 选择 `exe(local)` → 安装
3. 重启电脑

### 验证
```bash
nvidia-smi              # 看驱动版本
nvcc --version          # 看 CUDA 版本
```

> ⚠️ CUDA 版本不能高于 nvidia-smi 显示的驱动支持的版本

## 第三步：安装 PyTorch（CUDA 版）

```bash
# 千万不要 pip install torch（这是 CPU 版）

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8（老显卡）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 验证 PyTorch 能用 GPU
```python
import torch
print(torch.cuda.is_available())  # 必须输出 True
print(torch.cuda.get_device_name(0))  # 显示你的显卡名
```

如果输出 `False`：
1. 检查 CUDA 版本是否和 PyTorch 匹配
2. 重启终端/电脑
3. 重装 PyTorch（不要用缓存 pip install --no-cache-dir）

## 第四步：安装模型训练室

```bash
git clone https://github.com/ZBcxy/-model-training-room.git
cd model-training-room
python auto_setup.py   # 自动检测并安装所有依赖
```

**auto_setup.py 会自动安装**：
- LLaMA-Factory（训练引擎）
- llama.cpp（GGUF 导出）
- Ollama（本地推理，可选）

## 常见问题

### Q: CUDA out of memory（显存不足）
```
方案1：用 QLoRA 代替 LoRA（省 60% 显存）
方案2：减小 batch_size（减半 = 减 ~30% 显存）
方案3：减小 max_seq_length（2048 → 1024 = 减 ~40% 显存）
方案4：换更小的模型
```

### Q: bitsandbytes 装不上
```
pip install bitsandbytes 在某些系统上有问题。
解决：用 LoRA（非 QLoRA），不需要 bitsandbytes。
```

### Q: 训练速度太慢
```
- 确保 PyTorch 是 CUDA 版（不是 CPU 版）
- 检查 nvidia-smi 确认 GPU 在工作
- batch_size 调大（只要不 OOM）
- 用 BF16 代替 FP32
```

### Q: 笔记本能跑吗？
```
能。消费级 GPU 完全够用：
- RTX 3060 (12GB)：LoRA 微调 1.5B，流畅
- RTX 4060 (8GB)：QLoRA 微调 7B
- MacBook M 系列：用 MLX 框架（需额外配置）
```

## 推荐配置

| 预算 | 显卡 | 能做什么 |
|------|------|----------|
| ¥2,000 | RTX 3060 12GB 二手 | QLoRA 7B / LoRA 1.5B |
| ¥4,000 | RTX 4060 Ti 16GB | LoRA 7B |
| ¥8,000 | RTX 4070 Ti Super 16GB | LoRA 13B |
| ¥15,000 | RTX 4090 24GB | LoRA 30B / 全参 7B |
| 云端 | AutoDL 租 A100 | 随便玩 |
