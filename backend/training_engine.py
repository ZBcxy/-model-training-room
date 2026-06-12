"""
训练引擎：封装 LLaMA-Factory，提供参数推荐、配置文件生成、训练执行

这是模型训练室的核心模块。
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml  # type: ignore

from .hardware_checker import (
    VRAMBudget,
    calculate_vram_budget,
    check_training_feasibility,
    recommend_method,
)

# ============================================================
# Configuration
# ============================================================

EXPERIMENTS_DIR = Path(__file__).parent.parent / "data" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class TrainingConfig:
    """微调训练的完整配置"""
    # Model
    model_name_or_path: str = ""
    model_id: str = ""

    # Method
    finetuning_type: str = "lora"  # "lora" | "qlora" | "full"

    # Data
    dataset_path: str = ""
    dataset_format: str = "sharegpt"  # "sharegpt" | "alpaca"
    chat_template: str = ""
    max_seq_length: int = 2048

    # LoRA Hyperparameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target: str = "all"

    # Training Hyperparameters
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    optim: str = "adamw_8bit"

    # Optimization
    gradient_checkpointing: bool = True
    flash_attn: bool = True
    use_unsloth: bool = False

    # Output
    output_dir: str = ""
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3

    # VRAM
    vram_budget: dict = field(default_factory=dict)

    def to_llama_factory_config(self) -> dict:
        """转换为 LLaMA-Factory 可用的配置字典"""
        return {
            "model_name_or_path": self.model_name_or_path,
            "finetuning_type": self.finetuning_type,
            "dataset": "custom_dataset",  # Will be registered
            "dataset_format": self.dataset_format,
            "chat_template": self.chat_template or "auto",
            "max_seq_length": self.max_seq_length,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target": self.lora_target,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "optim": self.optim,
            "gradient_checkpointing": self.gradient_checkpointing,
            "flash_attn": "auto" if self.flash_attn else "disabled",
            "output_dir": self.output_dir,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
        }

    def to_dict(self) -> dict:
        """序列化为普通字典（用于 JSON 存储）"""
        return {
            "model_name_or_path": self.model_name_or_path,
            "model_id": self.model_id,
            "finetuning_type": self.finetuning_type,
            "dataset_path": self.dataset_path,
            "dataset_format": self.dataset_format,
            "chat_template": self.chat_template,
            "max_seq_length": self.max_seq_length,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target": self.lora_target,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "optim": self.optim,
            "gradient_checkpointing": self.gradient_checkpointing,
            "flash_attn": self.flash_attn,
            "output_dir": self.output_dir,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================
# Smart Parameter Recommendation
# ============================================================

PRESET_SCHEMES = {
    "quick": {
        "name": "⚡ 快速尝试",
        "description": "快速验证微调效果，约 1 小时",
        "num_train_epochs": 1,
        "lora_rank": 8,
        "lora_alpha": 16,
        "learning_rate": 5e-4,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 1024,
        "save_steps": 500,
    },
    "standard": {
        "name": "🎯 标准微调",
        "description": "平衡质量与速度，约 3-5 小时",
        "num_train_epochs": 3,
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "save_steps": 100,
    },
    "deep": {
        "name": "🔬 深度训练",
        "description": "追求最佳效果，约 8-12 小时",
        "num_train_epochs": 5,
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 4096,
        "save_steps": 50,
    },
}


def get_preset_scheme(preset: str) -> dict:
    """获取预设训练方案"""
    return PRESET_SCHEMES.get(preset, PRESET_SCHEMES["standard"]).copy()


def get_smart_recommendations(
    model_params_b: float,
    available_vram_gb: float,
    dataset_size: int,
) -> dict:
    """
    根据模型大小、显存和数据量，智能推荐所有参数。

    Returns:
        {
            "finetuning_type": str,
            "lora_rank": int,
            "lora_alpha": int,
            "learning_rate": float,
            "num_train_epochs": int,
            "per_device_train_batch_size": int,
            "gradient_accumulation_steps": int,
            "max_seq_length": int,
        }
    """
    # 1. Determine finetuning type
    method_rec = recommend_method(available_vram_gb)
    finetuning_type = method_rec["recommended"]
    if finetuning_type == "insufficient":
        finetuning_type = "qlora"  # Fallback

    # 2. LoRA rank based on model size
    if model_params_b <= 2:
        lora_rank = 8
        lora_alpha = 16
    elif model_params_b <= 8:
        lora_rank = 16
        lora_alpha = 32
    elif model_params_b <= 15:
        lora_rank = 32
        lora_alpha = 64
    else:
        lora_rank = 64
        lora_alpha = 128

    # 3. Batch size based on VRAM
    if available_vram_gb < 8:
        per_device_train_batch_size = 1
        gradient_accumulation_steps = 8
    elif available_vram_gb < 16:
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 4
    elif available_vram_gb < 24:
        per_device_train_batch_size = 4
        gradient_accumulation_steps = 4
    else:
        per_device_train_batch_size = 8
        gradient_accumulation_steps = 2

    # 4. Epochs based on dataset size
    if dataset_size < 1000:
        num_train_epochs = 10
    elif dataset_size < 5000:
        num_train_epochs = 5
    elif dataset_size < 20000:
        num_train_epochs = 3
    else:
        num_train_epochs = 2

    # 5. Learning rate
    if finetuning_type == "qlora":
        learning_rate = 2e-4
    elif finetuning_type == "lora":
        learning_rate = 2e-4
    else:
        learning_rate = 5e-5

    # 6. Max sequence length based on VRAM
    if available_vram_gb < 8:
        max_seq_length = 512
    elif available_vram_gb < 12:
        max_seq_length = 1024
    elif available_vram_gb < 24:
        max_seq_length = 2048
    else:
        max_seq_length = 4096

    return {
        "finetuning_type": finetuning_type,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_seq_length": max_seq_length,
    }


# ============================================================
# Chat Template Management
# ============================================================

# Known chat templates by model family
CHAT_TEMPLATE_MAP = {
    "qwen": "chatml",
    "qwen2": "chatml",
    "qwen2.5": "chatml",
    "llama": "llama3",
    "llama2": "llama2",
    "llama3": "llama3",
    "llama3.1": "llama3",
    "llama3.2": "llama3",
    "mistral": "mistral",
    "mixtral": "mistral",
    "gemma": "gemma",
    "gemma2": "gemma",
    "phi": "phi",
    "phi3": "phi3",
    "phi4": "phi4",
    "deepseek": "deepseek",
    "deepseekv2": "deepseek",
    "deepseekv3": "deepseek",
    "yi": "chatml",
    "baichuan": "baichuan",
    "chatglm": "chatglm3",
    "internlm": "intern2",
    "internlm2": "intern2",
}

CHAT_TEMPLATE_EXAMPLES = {
    "chatml": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>",
    "llama3": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
    "mistral": "[INST] {user_content} [/INST] {assistant_content}",
    "gemma": "<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n{content}<end_of_turn>",
    "phi3": "<|user|>\n{content}<|end|>\n<|assistant|>\n{content}<|end|>",
    "deepseek": "User: {user_content}\n\nAssistant: {assistant_content}",
}


def detect_chat_template(model_id: str, model_config: dict | None = None) -> dict:
    """
    自动检测模型的 Chat Template。

    Args:
        model_id: 模型 ID
        model_config: 模型的 config.json 内容（可选）

    Returns:
        {
            "template_name": str,
            "confidence": "high" | "medium" | "low",
            "source": "model_id" | "config" | "unknown",
        }
    """
    model_id_lower = model_id.lower()

    # Try from local config first
    if model_config:
        config_template = model_config.get("chat_template", "")
        if config_template:
            # Try to match known templates by signature
            template_sig = config_template[:50].lower()
            for name, example in CHAT_TEMPLATE_EXAMPLES.items():
                example_sig = example[:30].lower()
                if template_sig[:30] == example_sig[:30]:
                    return {"template_name": name, "confidence": "high", "source": "config"}

    # Match by model ID pattern
    for pattern, template in CHAT_TEMPLATE_MAP.items():
        if pattern in model_id_lower:
            return {"template_name": template, "confidence": "high", "source": "model_id"}

    # Heuristic based on common patterns
    if "chat" in model_id_lower or "instruct" in model_id_lower:
        return {"template_name": "chatml", "confidence": "low", "source": "heuristic"}
    if "llama" in model_id_lower:
        return {"template_name": "llama3", "confidence": "medium", "source": "heuristic"}

    return {"template_name": "chatml", "confidence": "low", "source": "unknown"}


# ============================================================
# Training Execution
# ============================================================

class TrainingExecutor:
    """
    训练执行器。

    管理和执行一次完整的微调训练任务。
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.experiment_id = f"exp_{int(time.time())}_{config.model_id.replace('/', '-')[:30]}"
        self.output_dir = EXPERIMENTS_DIR / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir = str(self.output_dir)

        self._process = None
        self._status = "initialized"  # initialized | running | paused | completed | failed | cancelled
        self._log_file = self.output_dir / "training.log"

    @property
    def status(self) -> str:
        return self._status

    def generate_config_yaml(self) -> Path:
        """生成 LLaMA-Factory 可用的 YAML 配置文件"""
        config_dict = self.config.to_llama_factory_config()

        # Add dataset override
        config_dict["dataset_dir"] = str(Path(self.config.dataset_path).parent)
        config_dict["dataset"] = Path(self.config.dataset_path).stem

        yaml_path = self.output_dir / "train_config.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

        return yaml_path

    def start(self, llama_factory_path: str | None = None) -> dict:
        """
        启动训练。

        Args:
            llama_factory_path: LLaMA-Factory 安装路径

        Returns:
            {"success": bool, "experiment_id": str, "error": str}
        """
        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, ensure_ascii=False, indent=2)

        # Generate YAML for LLaMA-Factory
        yaml_path = self.generate_config_yaml()

        # Check if LLaMA-Factory is available
        from .env_config import get_llama_factory_path
        if llama_factory_path:
            factory_dir = Path(llama_factory_path)
        else:
            detected = get_llama_factory_path()
            factory_dir = detected or Path("LLaMA-Factory")

        train_script = factory_dir / "src" / "train.py"

        cmd = [
            sys.executable,
            str(train_script),
            str(yaml_path),
        ]

        try:
            self._status = "running"
            with open(self._log_file, "w") as log_f:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=str(factory_dir),
                )

            # Don't wait — training runs in background
            # Store PID for monitoring
            with open(self.output_dir / "pid.txt", "w") as f:
                f.write(str(self._process.pid))

            return {
                "success": True,
                "experiment_id": self.experiment_id,
                "pid": self._process.pid,
                "output_dir": str(self.output_dir),
                "error": "",
            }

        except FileNotFoundError:
            return {
                "success": False,
                "experiment_id": self.experiment_id,
                "error": f"未找到 LLaMA-Factory。请确保已安装：\n"
                         f"  git clone https://github.com/hiyouga/LLaMA-Factory.git\n"
                         f"  cd LLaMA-Factory && pip install -e .",
            }
        except Exception as e:
            self._status = "failed"
            return {
                "success": False,
                "experiment_id": self.experiment_id,
                "error": f"启动训练失败: {str(e)}",
            }

    def stop(self) -> dict:
        """停止训练"""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()

        self._status = "cancelled"

        # Save a note
        with open(self.output_dir / "status.txt", "w") as f:
            f.write("cancelled")

        return {"success": True, "status": "cancelled"}

    def pause(self) -> dict:
        """暂停训练（发送 SIGSTOP）"""
        if self._process and self._process.poll() is None:
            self._process.send_signal(2)  # SIGINT - triggers graceful save

        self._status = "paused"
        with open(self.output_dir / "status.txt", "w") as f:
            f.write("paused")

        return {"success": True, "status": "paused"}

    def get_logs(self, tail_lines: int = 100) -> str:
        """获取最近的训练日志"""
        if not self._log_file.exists():
            return ""

        with open(self._log_file, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        return "".join(lines[-tail_lines:])


def create_training_config(
    model_id: str,
    model_path: str,
    dataset_path: str,
    dataset_format: str = "sharegpt",
    finetuning_type: str = "lora",
    preset: str = "standard",
    custom_params: dict | None = None,
    available_vram_gb: float = 16.0,
    model_params_b: float = 7.0,
    dataset_size: int = 10000,
) -> TrainingConfig:
    """
    一站式创建训练配置。

    Args:
        model_id: 模型 ID
        model_path: 模型本地路径
        dataset_path: 数据集路径
        dataset_format: 数据集格式
        finetuning_type: 微调方式
        preset: 预设方案名（quick/standard/deep）
        custom_params: 自定义参数覆盖
        available_vram_gb: 可用显存
        model_params_b: 模型参数量
        dataset_size: 数据集大小
    """
    # Start with smart recommendations
    recs = get_smart_recommendations(model_params_b, available_vram_gb, dataset_size)

    # Apply preset
    preset_params = get_preset_scheme(preset)

    # Merge: preset overrides smart, custom overrides preset
    params = {**recs, **preset_params}
    if custom_params:
        params.update(custom_params)

    # Detect chat template
    chat_template_info = detect_chat_template(model_id)
    chat_template = chat_template_info["template_name"]

    # Calculate VRAM budget
    if finetuning_type == "auto":
        finetuning_type = recs["finetuning_type"]

    budget = calculate_vram_budget(
        model_params_b=model_params_b,
        method=finetuning_type,
        batch_size=params["per_device_train_batch_size"],
        max_seq_length=params["max_seq_length"],
    )

    config = TrainingConfig(
        model_name_or_path=model_path,
        model_id=model_id,
        finetuning_type=finetuning_type,
        dataset_path=dataset_path,
        dataset_format=dataset_format,
        chat_template=chat_template,
        max_seq_length=params["max_seq_length"],
        lora_rank=params["lora_rank"],
        lora_alpha=params["lora_alpha"],
        learning_rate=params["learning_rate"],
        num_train_epochs=params["num_train_epochs"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        vram_budget={
            "total_estimate_gb": budget.total_estimate_gb,
            "model_weight_gb": budget.model_weight_gb,
            "optimizer_state_gb": budget.optimizer_state_gb,
            "activation_gb": budget.activation_gb,
            "available_vram_gb": available_vram_gb,
            "is_feasible": budget.total_estimate_gb <= available_vram_gb * 0.85,
        },
    )

    return config


# ============================================================
# Training Config Validation
# ============================================================

def validate_training_config(config: TrainingConfig) -> dict:
    """
    验证训练配置是否完整和合理。

    Returns:
        {"valid": bool, "issues": list of str, "warnings": list of str}
    """
    issues = []
    warnings = []

    # Required fields
    if not config.model_name_or_path:
        issues.append("模型路径未设置")
    elif not os.path.exists(config.model_name_or_path):
        issues.append(f"模型路径不存在: {config.model_name_or_path}")

    if not config.dataset_path:
        issues.append("数据集路径未设置")
    elif not os.path.exists(config.dataset_path):
        issues.append(f"数据集路径不存在: {config.dataset_path}")

    # Parameter sanity checks
    if config.lora_rank <= 0 or config.lora_rank > 256:
        issues.append(f"LoRA rank 不合理: {config.lora_rank} (建议 4-128)")

    if config.learning_rate <= 0 or config.learning_rate > 1e-2:
        issues.append(f"学习率不合理: {config.learning_rate} (建议 1e-5 ~ 5e-4)")

    if config.num_train_epochs <= 0 or config.num_train_epochs > 100:
        issues.append(f"训练轮数不合理: {config.num_train_epochs} (建议 1-20)")

    if config.per_device_train_batch_size <= 0 or config.per_device_train_batch_size > 128:
        issues.append(f"批次大小不合理: {config.per_device_train_batch_size} (建议 1-16)")

    if config.max_seq_length < 128:
        issues.append(f"序列长度太短: {config.max_seq_length} (建议 ≥ 512)")
    if config.max_seq_length > 32768:
        warnings.append(f"序列长度很大 ({config.max_seq_length})，可能超出显存")

    # VRAM budget warning
    budget = config.vram_budget
    if budget and not budget.get("is_feasible", True):
        issues.append(
            f"⚠️ 预估显存 ({budget.get('total_estimate_gb', '?')} GB) 超出可用显存。"
            f"建议：降低 batch size、使用 QLoRA、或减小 max_seq_length"
        )

    # Chat template warning
    if not config.chat_template:
        warnings.append("未设置 Chat Template，可能影响对话效果")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
    }


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 训练参数智能推荐")
    print("=" * 60)

    # Test: RTX 3060 (12GB) + 7B model + 10K dataset
    recs = get_smart_recommendations(
        model_params_b=7.0,
        available_vram_gb=11.6,
        dataset_size=10000,
    )
    print(f"  模型: 7B · 显存: 11.6GB · 数据: 10,000 条")
    for k, v in recs.items():
        print(f"    {k}: {v}")

    print()
    print("=" * 60)
    print("📋 预设方案")
    print("=" * 60)
    for name, scheme in PRESET_SCHEMES.items():
        print(f"  {scheme['name']}: {scheme['description']}")
        print(f"    epochs={scheme['num_train_epochs']} rank={scheme['lora_rank']} lr={scheme['learning_rate']}")

    print()
    print("=" * 60)
    print("💬 Chat Template 检测")
    print("=" * 60)
    for model_id in ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-v0.1"]:
        result = detect_chat_template(model_id)
        print(f"  {model_id}")
        print(f"    → {result['template_name']} (confidence: {result['confidence']})")

    print()
    print("=" * 60)
    print("✅ 训练引擎模块自检完成")
