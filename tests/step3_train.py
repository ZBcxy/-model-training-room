"""
实操验证 Step 3: 用真实模型+数据跑 QLoRA 微调

使用 LLaMA-Factory 对 Qwen2.5-1.5B 做 QLoRA 微调
500 条中文对话数据，1 epoch，验证训练链路

用法：
  python tests/step3_train.py
  # 或者用 LLaMA-Factory 的 CLI:
  # llamafactory-cli train tests/train_config.yaml
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply Python 3.14 compatibility fix BEFORE importing anything else
import tests.fix_py314_pickle

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "Qwen--Qwen2.5-1.5B-Instruct"
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "demo-zh-conversation" / "data.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "demo-qwen1.5b-qlora"
LLAMA_FACTORY_PATH = PROJECT_ROOT / "LLaMA-Factory"

# Training config
CONFIG = {
    "model_name_or_path": str(MODEL_PATH),
    "finetuning_type": "lora",
    "template": "qwen",
    "dataset": "demo_zh",
    "dataset_dir": str(DATASET_PATH.parent),
    "cutoff_len": 1024,
    "preprocessing_num_workers": 4,
    "output_dir": str(OUTPUT_DIR),
    "logging_steps": 5,
    "save_steps": 100,
    "plot_loss": True,
    "overwrite_output_dir": True,
    "do_train": True,
    # LoRA
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target": "all",
    # Training
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "optim": "adamw_torch",
    # Efficiency
    "gradient_checkpointing": True,
    # Reporting
    "report_to": "none",
    "bf16": True,
    # LoRA+
    "use_unsloth": False,
}


def check_prerequisites():
    """检查所有前置条件"""
    print("=" * 60)
    print("实操验证 Step 3: QLoRA 微调")
    print("=" * 60)
    print()

    ok = True

    # Model
    safetensors = list(MODEL_PATH.glob("*.safetensors"))
    if not safetensors:
        print("❌ 模型文件未找到。请先运行 step1_download_model.py")
        ok = False
    else:
        size = sum(f.stat().st_size for f in MODEL_PATH.rglob("*") if f.is_file())
        print(f"✅ 模型就绪: {MODEL_PATH} ({size / (1024**3):.1f} GB)")

    # Dataset
    if not DATASET_PATH.exists():
        print("❌ 数据集未找到。请先运行 step2_generate_data.py")
        ok = False
    else:
        with open(DATASET_PATH) as f:
            records = len(json.load(f))
        print(f"✅ 数据集就绪: {DATASET_PATH} ({records} 条)")

    # LLaMA-Factory
    train_script = LLAMA_FACTORY_PATH / "src" / "train.py"
    if not train_script.exists():
        print("❌ LLaMA-Factory 未找到")
        ok = False
    else:
        print(f"✅ LLaMA-Factory 就绪: {train_script}")

    # GPU
    import torch
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        ok = False
    else:
        gpu = torch.cuda.get_device_properties(0)
        print(f"✅ GPU: {gpu.name} ({gpu.total_memory / (1024**3):.1f} GB)")

    # Disk
    disk = os.statvfs(str(PROJECT_ROOT))
    free_gb = (disk.f_frsize * disk.f_bavail) / (1024**3)
    print(f"✅ 磁盘可用: {free_gb:.1f} GB")

    return ok


def register_dataset():
    """
    在 LLaMA-Factory 的 dataset_info.json 中注册我们的数据集。
    或者直接通过文件路径方式避免修改 LLaMA-Factory 的配置。
    """
    # 使用 LLaMA-Factory 的 "--dataset_dir" 参数，
    # 将我们的数据目录作为自定义数据集目录。
    # 需要在 dataset_dir 下有一个 dataset_info.json。
    dataset_info = {
        "demo_zh": {
            "file_name": "data.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            }
        }
    }

    info_path = DATASET_PATH.parent / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"   📋 数据集已注册: {info_path}")


def generate_train_config():
    """生成 LLaMA-Factory 训练配置 YAML"""
    import yaml

    config_path = OUTPUT_DIR / "train_config.yaml"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(CONFIG, f, default_flow_style=False, allow_unicode=True)

    print(f"   📋 训练配置已生成: {config_path}")
    return config_path


def run_training(config_path: Path):
    """执行训练"""
    print()
    print("=" * 60)
    print("🚀 开始训练...")
    print("=" * 60)

    train_script = LLAMA_FACTORY_PATH / "src" / "train.py"

    cmd = [
        sys.executable,
        str(train_script),
        str(config_path),
    ]

    print(f"   命令: {' '.join(cmd)}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print()

    import subprocess

    # Apply pickle fix in subprocess via PYTHONSTARTUP
    env = os.environ.copy()
    fix_script = PROJECT_ROOT / "tests" / "fix_py314_pickle.py"
    env["PYTHONSTARTUP"] = str(fix_script)
    # Also set HF endpoint mirror in case network is slow
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    start = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(LLAMA_FACTORY_PATH),
        env=env,
    )

    elapsed = time.time() - start

    if result.returncode == 0:
        print()
        print("=" * 60)
        print(f"✅ 训练完成！")
        print(f"   耗时: {elapsed:.0f} 秒 ({elapsed/60:.1f} 分钟)")
        print(f"   输出: {OUTPUT_DIR}")
        print("=" * 60)
        return True
    else:
        print()
        print("=" * 60)
        print(f"❌ 训练失败 (exit code: {result.returncode})")
        print("=" * 60)
        return False


def find_checkpoint():
    """找到最新的 checkpoint"""
    checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"))
    if checkpoints:
        return checkpoints[-1]
    # Sometimes LLaMA-Factory saves to a different pattern
    adapter = OUTPUT_DIR / "adapter_model"
    if adapter.exists():
        return adapter
    return None


if __name__ == "__main__":
    # 1. Check
    if not check_prerequisites():
        print()
        print("❌ 前置条件不满足，请先完成前面的步骤。")
        sys.exit(1)

    # 2. Register dataset
    print()
    print("─" * 60)
    register_dataset()

    # 3. Generate config
    config_path = generate_train_config()

    # 4. Train
    print()
    print("─" * 60)
    print("💡 显存预估: QLoRA + 1.5B 模型 + batch_size=2")
    print("   预估显存占用: ~4GB (你的 RTX 3060 12GB 绰绰有余)")
    print()

    success = run_training(config_path)

    # 5. Find checkpoint
    if success:
        ckpt = find_checkpoint()
        if ckpt:
            print(f"   💾 Checkpoint: {ckpt}")
            # Save as latest for export step
            latest_link = OUTPUT_DIR / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(ckpt.relative_to(OUTPUT_DIR))
            print(f"   🔗 最新: {latest_link} -> {ckpt.name}")

    sys.exit(0 if success else 1)
