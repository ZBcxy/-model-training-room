"""
训练监控：实时 Loss 曲线、GPU 状态、日志解析

独立于训练进程运行，通过读取日志文件和系统 API 监控训练状态。
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil
import torch


# ============================================================
# Data Classes
# ============================================================

@dataclass
class TrainingMetrics:
    """从训练日志中解析的指标"""
    step: int = 0
    epoch: float = 0.0
    loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    tokens_per_second: float = 0.0
    elapsed_seconds: float = 0.0
    timestamp: float = 0.0


@dataclass
class GPUStatus:
    """GPU 实时状态"""
    index: int = 0
    name: str = ""
    utilization_pct: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    temperature_c: float = 0.0
    power_w: float = 0.0


@dataclass
class TrainingProgress:
    """训练进度快照"""
    experiment_id: str = ""
    status: str = "unknown"  # running | paused | completed | failed
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    best_loss: float = float("inf")
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    tokens_per_second: float = 0.0
    gpu_status: list[GPUStatus] = field(default_factory=list)
    metrics_history: list[dict] = field(default_factory=list)
    sample_outputs: list[dict] = field(default_factory=list)
    log_tail: str = ""


# ============================================================
# GPU Monitoring
# ============================================================

def get_gpu_status() -> list[GPUStatus]:
    """获取当前 GPU 实时状态"""
    gpus = []

    if not torch.cuda.is_available():
        return gpus

    try:
        import pynvml
        pynvml.nvmlInit()

        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_pct = float(util.gpu)
            except Exception:
                utilization_pct = 0.0

            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used_gb = mem.used / (1024 ** 3)
                mem_total_gb = mem.total / (1024 ** 3)
            except Exception:
                mem_used_gb = 0.0
                mem_total_gb = 0.0

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0.0

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power = 0.0

            gpus.append(GPUStatus(
                index=i,
                name=name.decode() if isinstance(name, bytes) else name,
                utilization_pct=utilization_pct,
                memory_used_gb=round(mem_used_gb, 2),
                memory_total_gb=round(mem_total_gb, 2),
                temperature_c=temp,
                power_w=round(power, 1),
            ))

    except ImportError:
        # Fallback: use torch only
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024 ** 3)
            try:
                free_mb, _ = torch.cuda.mem_get_info(i)
                used_gb = (props.total_memory - free_mb) / (1024 ** 3)
            except Exception:
                used_gb = 0.0

            gpus.append(GPUStatus(
                index=i,
                name=props.name,
                memory_used_gb=round(used_gb, 2),
                memory_total_gb=round(total_gb, 2),
            ))

    return gpus


# ============================================================
# Log Parsing
# ============================================================

# LLaMA-Factory log patterns
LOG_PATTERNS = {
    "step_loss": re.compile(
        r"\{.*'loss':\s*([\d.]+).*'learning_rate':\s*([\deE\-.]+).*'epoch':\s*([\d.]+)"
    ),
    "progress": re.compile(
        r"(\d+)%\|.*\|.*/(\d+)\s*\[.*<.*,\s*([\d.]+)it/s"
    ),
    "train_loss": re.compile(
        r"\{.*'loss':\s*([\d.]+)"
    ),
    "step_marker": re.compile(
        r"\{.*'step':\s*(\d+)"
    ),
}


def parse_training_log(log_path: str, tail_lines: int = 500) -> list[TrainingMetrics]:
    """
    解析训练日志，提取指标历史。

    Returns:
        List of TrainingMetrics, one per logged step.
    """
    if not os.path.exists(log_path):
        return []

    metrics_list = []

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        lines = lines[-tail_lines:]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to parse as JSON (LLaMA-Factory logs training info as JSON)
            try:
                if line.startswith("{") and line.endswith("}"):
                    data = json.loads(line)
                    if "loss" in data:
                        metrics = TrainingMetrics(
                            step=data.get("step", 0),
                            epoch=data.get("epoch", 0.0),
                            loss=data.get("loss", 0.0),
                            learning_rate=data.get("learning_rate", 0.0),
                            grad_norm=data.get("grad_norm", 0.0),
                            elapsed_seconds=data.get("elapsed_seconds", 0.0),
                            timestamp=time.time(),
                        )
                        metrics_list.append(metrics)
                        continue
            except json.JSONDecodeError:
                pass

            # Try regex patterns as fallback
            match = LOG_PATTERNS["step_loss"].search(line)
            if match:
                metrics = TrainingMetrics(
                    loss=float(match.group(1)),
                    learning_rate=float(match.group(2)),
                    epoch=float(match.group(3)),
                    timestamp=time.time(),
                )
                metrics_list.append(metrics)

    except Exception as e:
        print(f"[Monitor] Error parsing log: {e}")

    return metrics_list


def get_training_progress(
    experiment_dir: str,
    total_steps: int = 0,
    total_epochs: int = 0,
) -> TrainingProgress:
    """
    获取训练的当前进度快照。
    """
    exp_dir = Path(experiment_dir)
    log_file = exp_dir / "training.log"
    config_file = exp_dir / "config.json"
    status_file = exp_dir / "status.txt"

    progress = TrainingProgress(
        experiment_id=exp_dir.name,
    )

    # Check status
    if status_file.exists():
        progress.status = status_file.read_text().strip()
    else:
        progress.status = "running"

    # Read config for total steps
    if config_file.exists():
        try:
            with open(config_file) as f:
                cfg = json.load(f)
                progress.total_epochs = cfg.get("num_train_epochs", total_epochs)
        except Exception:
            pass

    progress.total_epochs = progress.total_epochs or total_epochs
    progress.total_steps = total_steps

    # Parse logs
    if log_file.exists():
        metrics = parse_training_log(str(log_file))
        if metrics:
            latest = metrics[-1]
            progress.current_step = latest.step
            progress.current_loss = latest.loss
            progress.elapsed_seconds = latest.elapsed_seconds
            progress.current_epoch = int(latest.epoch)

            best = min(m.loss for m in metrics)
            progress.best_loss = best

            # Simple token/s estimation from recent steps
            if len(metrics) >= 2:
                recent = metrics[-10:] if len(metrics) >= 10 else metrics[-2:]
                if len(recent) >= 2:
                    time_diff = recent[-1].elapsed_seconds - recent[0].elapsed_seconds
                    step_diff = recent[-1].step - recent[0].step
                    if time_diff > 0:
                        progress.tokens_per_second = step_diff / time_diff

            # Estimate remaining time
            if progress.current_step > 0 and progress.total_steps > 0:
                steps_remaining = progress.total_steps - progress.current_step
                if progress.tokens_per_second > 0:
                    progress.estimated_remaining_seconds = steps_remaining / progress.tokens_per_second

            progress.metrics_history = [
                {"step": m.step, "epoch": round(m.epoch, 2), "loss": round(m.loss, 4),
                 "lr": m.learning_rate, "time": m.elapsed_seconds}
                for m in metrics
            ]

    # GPU status
    progress.gpu_status = get_gpu_status()

    # Log tail
    if log_file.exists():
        try:
            with open(log_file, encoding="utf-8", errors="replace") as f:
                progress.log_tail = "".join(f.readlines()[-20:])
        except Exception:
            pass

    return progress


# ============================================================
# Checkpoint Management
# ============================================================

def list_checkpoints(experiment_dir: str) -> list[dict]:
    """列出训练实验的所有 checkpoint"""
    exp_dir = Path(experiment_dir)
    checkpoints = []

    for ckpt_dir in sorted(exp_dir.glob("checkpoint-*")):
        if not ckpt_dir.is_dir():
            continue

        step = int(ckpt_dir.name.split("-")[-1])

        # Calculate size
        size = 0
        for f in ckpt_dir.rglob("*"):
            if f.is_file():
                size += f.stat().st_size

        checkpoints.append({
            "path": str(ckpt_dir),
            "step": step,
            "size_mb": round(size / (1024 ** 2), 2),
            "modified": os.path.getmtime(str(ckpt_dir)),
        })

    # Sort by step
    checkpoints.sort(key=lambda x: x["step"], reverse=True)

    return checkpoints


# ============================================================
# Sample Output Extraction
# ============================================================

def get_sample_outputs(log_path: str, max_samples: int = 3) -> list[dict]:
    """
    从训练日志中提取最近的样本输出，用于实时查看训练效果。
    """
    if not os.path.exists(log_path):
        return []

    samples = []

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Look for evaluation/prediction sections
        # LLaMA-Factory logs predictions during eval
        eval_pattern = re.compile(
            r'predict_(\d+).*?"input":\s*"(.*?)".*?"predict":\s*"(.*?)".*?"label":\s*"(.*?)"',
            re.DOTALL,
        )

        matches = eval_pattern.finditer(content)
        for match in matches:
            samples.append({
                "step": match.group(1),
                "input": match.group(2)[:200],
                "predict": match.group(3)[:500],
                "label": match.group(4)[:500],
            })

    except Exception:
        pass

    return samples[-max_samples:]


# ============================================================
# Utility
# ============================================================

def format_duration(seconds: float) -> str:
    """将秒数格式化为人类可读的时间"""
    if seconds < 60:
        return f"{int(seconds)} 秒"
    elif seconds < 3600:
        return f"{int(seconds // 60)} 分 {int(seconds % 60)} 秒"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h} 小时 {m} 分"


def format_vram_bar(used_gb: float, total_gb: float, width: int = 20) -> str:
    """生成显存使用条"""
    ratio = min(used_gb / max(total_gb, 1), 1.0)
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} ({used_gb:.1f}/{total_gb:.1f} GB)"


if __name__ == "__main__":
    print("=" * 60)
    print("🖥️  GPU 状态")
    print("=" * 60)
    gpus = get_gpu_status()
    for gpu in gpus:
        print(f"  GPU {gpu.index}: {gpu.name}")
        print(f"    显存: {gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.1f} GB")
        print(f"    利用率: {gpu.utilization_pct}% · 温度: {gpu.temperature_c}°C")
        print(f"    {format_vram_bar(gpu.memory_used_gb, gpu.memory_total_gb)}")

    print()
    print(f"  时长格式化: {format_duration(3723)}")
    print(f"  时长格式化: {format_duration(125)}")
