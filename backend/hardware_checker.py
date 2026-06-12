"""
硬件环境检测与显存预算计算器

负责：
- GPU 型号/显存/CUDA 版本检测
- 系统内存/磁盘检测
- 环境依赖检查（PyTorch、CUDA toolkit）
- 显存预算计算（模型权重、优化器、激活值分项估算）
- 微调方式智能推荐
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

import psutil
import torch


# ============================================================
# Data Classes
# ============================================================

@dataclass
class GPUInfo:
    """单张 GPU 的信息"""
    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    vram_used_gb: float
    compute_capability: str = ""
    cuda_version: str = ""
    temperature_c: Optional[float] = None
    utilization_pct: Optional[float] = None


@dataclass
class SystemInfo:
    """系统硬件信息汇总"""
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_count: int = 0
    cpu_name: str = ""
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_path: str = ""
    python_version: str = ""
    pytorch_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    os_name: str = ""

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def total_vram_gb(self) -> float:
        return sum(g.vram_total_gb for g in self.gpus)

    @property
    def max_single_vram_gb(self) -> float:
        """单卡最大显存（用于判断能否跑某个模型）"""
        return max((g.vram_total_gb for g in self.gpus), default=0.0)


@dataclass
class VRAMBudget:
    """显存预算拆分明细"""
    model_params_b: int  # 模型参数量
    method: str  # qlora / lora / full
    precision: str  # 4bit / 8bit / bf16 / fp16

    model_weight_gb: float = 0.0
    optimizer_state_gb: float = 0.0
    activation_gb: float = 0.0
    data_gb: float = 0.0
    overhead_gb: float = 0.0

    total_estimate_gb: float = 0.0
    available_vram_gb: float = 0.0
    is_feasible: bool = True
    warning: str = ""


# ============================================================
# GPU Detection
# ============================================================

def get_gpu_info() -> list[GPUInfo]:
    """检测所有可用 GPU 的详细信息"""
    gpus = []

    if not torch.cuda.is_available():
        return gpus

    cuda_ver = torch.version.cuda or "unknown"

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        total_mb = props.total_memory
        total_gb = total_mb / (1024 ** 3)

        # 获取计算能力
        cc = f"{props.major}.{props.minor}"

        # 获取当前显存使用情况
        try:
            free_mb, total_mb_2 = torch.cuda.mem_get_info(i)
            free_gb = free_mb / (1024 ** 3)
            used_gb = (total_mb_2 - free_mb) / (1024 ** 3)
        except Exception:
            free_gb = total_gb
            used_gb = 0.0

        # 尝试通过 nvidia-ml-py 获取温度和利用率
        temp = None
        util = None
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                pass
        except Exception:
            pass

        gpus.append(GPUInfo(
            index=i,
            name=name,
            vram_total_gb=round(total_gb, 2),
            vram_free_gb=round(free_gb, 2),
            vram_used_gb=round(used_gb, 2),
            compute_capability=cc,
            cuda_version=cuda_ver,
            temperature_c=temp,
            utilization_pct=util,
        ))

    return gpus


# ============================================================
# System Detection
# ============================================================

def get_system_info() -> SystemInfo:
    """全面收集系统硬件和软件环境信息"""
    info = SystemInfo()

    # --- GPU ---
    info.gpus = get_gpu_info()

    # --- CPU ---
    info.cpu_count = psutil.cpu_count(logical=True)
    info.cpu_name = _get_cpu_name()

    # --- RAM ---
    ram = psutil.virtual_memory()
    info.ram_total_gb = round(ram.total / (1024 ** 3), 2)
    info.ram_available_gb = round(ram.available / (1024 ** 3), 2)

    # --- Disk ---
    # 默认检测当前工作目录所在的磁盘
    cwd = os.getcwd()
    disk = psutil.disk_usage(cwd)
    info.disk_total_gb = round(disk.total / (1024 ** 3), 2)
    info.disk_free_gb = round(disk.free / (1024 ** 3), 2)
    info.disk_path = cwd

    # --- Python / PyTorch ---
    info.python_version = sys.version.split()[0]
    info.pytorch_version = torch.__version__
    info.cuda_available = torch.cuda.is_available()
    info.cuda_version = torch.version.cuda or "N/A"
    info.os_name = f"{sys.platform}"

    return info


def _get_cpu_name() -> str:
    """获取 CPU 型号名称"""
    try:
        if sys.platform == "linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        elif sys.platform == "darwin":
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True)
            return result.stdout.strip()
    except Exception:
        pass
    return "Unknown CPU"


# ============================================================
# VRAM Budget Calculator
# ============================================================

def calculate_vram_budget(
    model_params_b: float,  # 模型参数量，单位 亿 (B)
    method: str = "lora",
    precision: str = "bf16",
    batch_size: int = 4,
    max_seq_length: int = 2048,
    gradient_checkpointing: bool = True,
    trainable_ratio: float = 0.01,  # LoRA 可训参数比例
) -> VRAMBudget:
    """
    计算微调所需的显存预算。

    Args:
        model_params_b: 模型参数量（B，十亿）
        method: 'qlora' | 'lora' | 'full'
        precision: '4bit' | '8bit' | 'bf16' | 'fp16'
        batch_size: 批次大小
        max_seq_length: 最大序列长度
        gradient_checkpointing: 是否启用梯度检查点
        trainable_ratio: LoRA 可训参数比例（默认 1%）
    """
    params = model_params_b * 1e9  # 转为实际参数量

    # --- 精度对应的每参数字节数 ---
    bytes_per_param = {
        "4bit": 0.5,
        "8bit": 1.0,
        "bf16": 2.0,
        "fp16": 2.0,
        "fp32": 4.0,
    }

    # --- 模型权重显存 ---
    if method == "qlora":
        model_bytes = bytes_per_param["4bit"]
        trainable_bytes = bytes_per_param["bf16"]  # LoRA 权重用 bf16
    elif method == "lora":
        model_bytes = bytes_per_param[precision]
        trainable_bytes = bytes_per_param[precision]
    else:  # full
        model_bytes = bytes_per_param[precision]
        trainable_bytes = bytes_per_param[precision]

    # 1. 模型权重（推理时的基础占用）
    model_weight_gb = (params * model_bytes) / (1024 ** 3)

    # 2. 优化器状态
    if method == "qlora":
        # QLoRA: 只有少量 LoRA 参数需要优化器状态
        trainable_params = params * trainable_ratio
        # AdamW: 参数 + 动量 + 方差 = 3x (bf16 存储)
        optimizer_state_gb = (trainable_params * bytes_per_param["bf16"] * 3) / (1024 ** 3)
    elif method == "lora":
        trainable_params = params * trainable_ratio
        optimizer_state_gb = (trainable_params * bytes_per_param["bf16"] * 3) / (1024 ** 3)
    else:  # full
        # 全参微调：所有参数都需要优化器状态
        # 使用 AdamW 的优化器状态 = 参数 * 3 (param + momentum + variance)
        optimizer_state_gb = (params * bytes_per_param[precision] * 3) / (1024 ** 3)

    # 3. 激活值（与 batch_size × seq_len 正相关）
    # 经验公式：约为 (hidden_dim * num_layers * batch_size * seq_len * 2) bytes
    # 简化估算：每个 token 大约产生 hidden_dim * num_layers * 2 bytes 的激活值
    hidden_dim = _estimate_hidden_dim(params)
    num_layers = _estimate_num_layers(params)
    activation_per_token = hidden_dim * num_layers * 2  # bytes (粗略)
    total_tokens_in_flight = batch_size * max_seq_length
    activation_gb = (activation_per_token * total_tokens_in_flight) / (1024 ** 3)

    # 梯度检查点可以减少约 60% 的激活显存
    if gradient_checkpointing:
        activation_gb *= 0.4

    # 4. 数据加载开销
    data_gb = (batch_size * max_seq_length * 2) / (1024 ** 3) * 3  # 输入+输出+中间缓冲

    # 5. 附加开销（框架、CUDA context 等）
    overhead_gb = model_weight_gb * 0.15 + 0.5

    total = model_weight_gb + optimizer_state_gb + activation_gb + data_gb + overhead_gb

    return VRAMBudget(
        model_params_b=model_params_b,
        method=method,
        precision=precision,
        model_weight_gb=round(model_weight_gb, 2),
        optimizer_state_gb=round(optimizer_state_gb, 2),
        activation_gb=round(activation_gb, 2),
        data_gb=round(data_gb, 2),
        overhead_gb=round(overhead_gb, 2),
        total_estimate_gb=round(total, 2),
    )


def _estimate_hidden_dim(params: float) -> int:
    """根据参数量大致估算 hidden dimension"""
    # 粗略估算: hidden_dim ≈ sqrt(params / num_layers / 4)
    num_layers = _estimate_num_layers(params)
    return max(1024, int((params / num_layers / 4) ** 0.5))


def _estimate_num_layers(params: float) -> int:
    """根据参数量大致估算层数"""
    if params < 2e9:
        return 24
    elif params < 8e9:
        return 32
    elif params < 15e9:
        return 40
    elif params < 40e9:
        return 60
    else:
        return 80


def recommend_method(available_vram_gb: float) -> dict:
    """
    根据可用显存推荐微调方式。

    Returns:
        dict with keys: recommended, options (list of feasible options)
    """
    options = []

    # QLoRA: 最低门槛 ~6GB
    if available_vram_gb >= 6:
        options.append({
            "method": "qlora",
            "label": "QLoRA (4-bit量化微调)",
            "min_vram": "6GB",
            "models_supported": "最高可微调 ~70B 模型",
            "speed": "中等",
            "quality": "良好",
            "icon": "⚡",
        })

    # LoRA: ~16GB
    if available_vram_gb >= 16:
        options.append({
            "method": "lora",
            "label": "LoRA (标准低秩微调)",
            "min_vram": "16GB",
            "models_supported": "最高可微调 ~13B 模型",
            "speed": "中等",
            "quality": "优秀 ⭐",
            "icon": "🎯",
        })

    # Full fine-tuning: ~40GB
    if available_vram_gb >= 40:
        options.append({
            "method": "full",
            "label": "全参数微调",
            "min_vram": "40GB",
            "models_supported": "最高可微调 ~7B 模型",
            "speed": "较慢",
            "quality": "最佳",
            "icon": "🏋️",
        })

    recommended = options[0]["method"] if options else "insufficient"

    return {
        "recommended": recommended,
        "available_vram_gb": available_vram_gb,
        "options": options,
        "suggestion": _get_vram_suggestion(available_vram_gb, options),
    }


def _get_vram_suggestion(vram_gb: float, options: list) -> str:
    """生成显存建议文案"""
    if not options:
        return ("⚠️ 显存不足（<6GB）。建议：\n"
                "  · 使用云端 GPU（如 AutoDL、Colab）\n"
                "  · 选择更小的模型（<1B 参数）\n"
                "  · 使用 Ollama 直接运行已微调的模型")
    elif vram_gb < 8:
        return ("💡 显存有限。建议：\n"
                "  · 使用 QLoRA + 小模型（≤3B）\n"
                "  · batch_size 设为 1\n"
                "  · 减小 max_seq_length 到 1024")
    elif vram_gb < 16:
        return ("💡 显存适中。建议：\n"
                "  · 使用 QLoRA 微调 7B 模型\n"
                "  · batch_size 设为 2-4")
    elif vram_gb < 24:
        return ("💡 显存充足。建议：\n"
                "  · 使用 LoRA 微调 7B-13B 模型\n"
                "  · 或 QLoRA 微调 30B+ 模型")
    else:
        return ("💡 显存充裕。建议：\n"
                "  · 可以使用 LoRA 微调大模型\n"
                "  · 或尝试全参微调 7B 模型")


def check_training_feasibility(
    model_params_b: float,
    method: str,
    available_vram_gb: float,
    batch_size: int = 4,
    max_seq_length: int = 2048,
) -> VRAMBudget:
    """
    检查在给定硬件上微调某个模型是否可行。

    Returns:
        VRAMBudget with is_feasible 和 warning 字段填充
    """
    precision_map = {
        "qlora": "4bit",
        "lora": "bf16",
        "full": "bf16",
    }
    precision = precision_map.get(method, "bf16")

    budget = calculate_vram_budget(
        model_params_b=model_params_b,
        method=method,
        precision=precision,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )

    budget.available_vram_gb = available_vram_gb
    budget.is_feasible = budget.total_estimate_gb <= available_vram_gb * 0.85  # 留 15% 余量

    if not budget.is_feasible:
        shortage = budget.total_estimate_gb - available_vram_gb * 0.85
        budget.warning = (
            f"⚠️ 预估显存 {budget.total_estimate_gb:.1f}GB 超出可用显存 "
            f"{available_vram_gb:.1f}GB（含15%安全余量）。\n"
            f"建议：① 降低 batch size ② 使用 QLoRA ③ 减小 max_seq_length"
        )

    return budget


# ============================================================
# Environment Dependency Check
# ============================================================

def check_environment_deps() -> dict:
    """
    检查关键依赖是否就绪。

    Returns:
        dict: {package_name: {"installed": bool, "version": str, "required": str}}
    """
    deps = {}

    # PyTorch
    deps["torch"] = {
        "installed": True,
        "version": torch.__version__,
        "required": ">=2.1.0",
    }

    # CUDA
    deps["cuda"] = {
        "installed": torch.cuda.is_available(),
        "version": torch.version.cuda or "N/A",
        "required": "推荐 CUDA 11.8+",
    }

    # Transformers
    try:
        import transformers
        deps["transformers"] = {
            "installed": True,
            "version": transformers.__version__,
            "required": ">=4.41.0",
        }
    except ImportError:
        deps["transformers"] = {"installed": False, "version": "N/A", "required": ">=4.41.0"}

    # PEFT
    try:
        import peft
        deps["peft"] = {
            "installed": True,
            "version": peft.__version__,
            "required": ">=0.10.0",
        }
    except ImportError:
        deps["peft"] = {"installed": False, "version": "N/A", "required": ">=0.10.0"}

    # datasets
    try:
        import datasets
        deps["datasets"] = {
            "installed": True,
            "version": datasets.__version__,
            "required": ">=2.18.0",
        }
    except ImportError:
        deps["datasets"] = {"installed": False, "version": "N/A", "required": ">=2.18.0"}

    # llama.cpp (可选)
    try:
        import llama_cpp
        deps["llama.cpp"] = {
            "installed": True,
            "version": llama_cpp.__version__,
            "required": ">=0.2.75 (用于GGUF导出)",
        }
    except ImportError:
        deps["llama.cpp"] = {"installed": False, "version": "N/A", "required": ">=0.2.75 (用于GGUF导出)"}

    # LLaMA-Factory (可选，运行时检查)
    deps["llama-factory"] = {
        "installed": _check_llama_factory(),
        "version": _get_llama_factory_version(),
        "required": "训练引擎，强烈推荐安装",
    }

    return deps


def _check_llama_factory() -> bool:
    """检查 LLaMA-Factory 是否可用"""
    try:
        # 尝试导入 LLaMA-Factory 的核心模块
        # LLaMA-Factory 需要从 GitHub 安装:
        # git clone https://github.com/hiyouga/LLaMA-Factory.git
        # pip install -e LLaMA-Factory/
        import importlib
        importlib.import_module("llamafactory")
        return True
    except ImportError:
        return False


def _get_llama_factory_version() -> str:
    """获取 LLaMA-Factory 版本"""
    try:
        import llamafactory
        return getattr(llamafactory, "__version__", "unknown")
    except Exception:
        return "未安装"


def generate_env_report() -> str:
    """
    生成人类可读的环境报告文本。

    用于在界面上展示给用户。
    """
    info = get_system_info()
    deps = check_environment_deps()
    rec = recommend_method(info.max_single_vram_gb)

    lines = []
    lines.append("=" * 60)
    lines.append("🖥️  模型训练室 · 环境检测报告")
    lines.append("=" * 60)
    lines.append("")

    # GPU
    lines.append("─" * 60)
    lines.append("🎮 GPU 信息")
    lines.append("─" * 60)
    if info.has_gpu:
        for gpu in info.gpus:
            lines.append(f"  GPU {gpu.index}: {gpu.name}")
            lines.append(f"    显存: {gpu.vram_total_gb:.1f} GB (可用 {gpu.vram_free_gb:.1f} GB)")
            lines.append(f"    CUDA: {gpu.cuda_version} · 计算能力 {gpu.compute_capability}")
            if gpu.temperature_c:
                lines.append(f"    温度: {gpu.temperature_c}°C · 利用率: {gpu.utilization_pct}%")
    else:
        lines.append("  ❌ 未检测到 GPU（仅支持 CPU 推理）")

    # System
    lines.append("")
    lines.append("─" * 60)
    lines.append("💻 系统信息")
    lines.append("─" * 60)
    lines.append(f"  CPU: {info.cpu_name} ({info.cpu_count} 核心)")
    lines.append(f"  内存: 总计 {info.ram_total_gb:.1f} GB · 可用 {info.ram_available_gb:.1f} GB")
    lines.append(f"  磁盘: 总计 {info.disk_total_gb:.1f} GB · 可用 {info.disk_free_gb:.1f} GB")
    lines.append(f"  Python: {info.python_version} · PyTorch: {info.pytorch_version}")
    lines.append(f"  CUDA 可用: {'✅' if info.cuda_available else '❌'}")

    # Recommendation
    lines.append("")
    lines.append("─" * 60)
    lines.append("🎯 微调能力评估")
    lines.append("─" * 60)
    if info.has_gpu:
        lines.append(f"  可用显存: {info.max_single_vram_gb:.1f} GB")
        lines.append(f"  推荐方式: {rec['recommended'].upper()}")
        for opt in rec["options"]:
            lines.append(f"    {opt['icon']} {opt['label']} — {opt['models_supported']}")
    lines.append("")
    lines.append(rec["suggestion"])

    lines.append("")
    lines.append("─" * 60)
    lines.append("📦 依赖检查")
    lines.append("─" * 60)
    for name, dep in deps.items():
        status = "✅" if dep["installed"] else "❌"
        lines.append(f"  {status} {name}: {dep['version']} (需要 {dep['required']})")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================
# CLI Entry Point (for testing standalone)
# ============================================================

if __name__ == "__main__":
    print(generate_env_report())
