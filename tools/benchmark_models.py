#!/usr/bin/env python3
"""
多模型对比训练 & 评测框架

用法: python tools/benchmark_models.py --data data.json --models qwen1.5b,llama3.2-3b

用同一份数据依次训练多个模型，生成对比报告。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

# 预设的对比模型配置
PRESET_MODELS = {
    "qwen1.5b": {
        "name": "Qwen2.5-1.5B-Instruct",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "source": "modelscope",
        "params_b": 1.5,
        "vram_qlora_gb": 4,
        "vram_lora_gb": 6,
    },
    "qwen7b": {
        "name": "Qwen2.5-7B-Instruct",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "source": "modelscope",
        "params_b": 7.0,
        "vram_qlora_gb": 6,
        "vram_lora_gb": 16,
    },
    "llama3.2-3b": {
        "name": "Llama-3.2-3B-Instruct",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "source": "huggingface",
        "params_b": 3.0,
        "vram_qlora_gb": 4,
        "vram_lora_gb": 8,
    },
    "chatglm3-6b": {
        "name": "ChatGLM3-6B",
        "model_id": "ZhipuAI/chatglm3-6b",
        "source": "modelscope",
        "params_b": 6.0,
        "vram_qlora_gb": 6,
        "vram_lora_gb": 14,
    },
}


def download_model(model_config):
    """下载模型"""
    model_dir = PROJECT / "data/models" / model_config["model_id"].replace("/", "--")
    if (model_dir / "model.safetensors").exists():
        print(f"  ✅ 已存在，跳过下载")
        return str(model_dir)

    print(f"  📥 下载中...")
    from modelscope import snapshot_download
    path = snapshot_download(model_config["model_id"], cache_dir=str(model_dir))
    return path


def train_model(model_config, data_path, output_dir):
    """训练单个模型"""
    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect best method based on VRAM
    from backend.hardware_checker import get_system_info
    info = get_system_info()
    vram = info.max_single_vram_gb if info.has_gpu else 8.0

    if vram >= model_config["vram_lora_gb"]:
        method = "lora"
    else:
        method = "qlora"

    # Detect template
    from backend.model_cards import match_card
    card = match_card(model_config["model_id"])
    template = card["chat_template"] if card else "chatml"

    config = {
        "model_name_or_path": str(PROJECT / "data/models" / model_config["model_id"].replace("/", "--")),
        "finetuning_type": method,
        "template": template,
        "dataset": "benchmark_data",
        "dataset_dir": str(Path(data_path).parent),
        "cutoff_len": 2048,
        "output_dir": str(output_dir),
        "logging_steps": 10,
        "save_steps": 500,
        "overwrite_output_dir": True,
        "do_train": True,
        "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4, "num_train_epochs": 2,
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
        "optim": "adamw_torch", "gradient_checkpointing": True,
        "report_to": "none", "bf16": True,
    }

    yaml_path = output_dir / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, allow_unicode=True)

    print(f"  🚀 训练: {method} | {template}")
    start = time.time()
    r = subprocess.run([
        sys.executable, str(PROJECT / "LLaMA-Factory/src/train.py"), str(yaml_path)
    ])
    elapsed = time.time() - start

    if r.returncode != 0:
        return {"success": False, "error": f"Exit code {r.returncode}"}

    # Read final loss
    log_file = output_dir / "trainer_log.jsonl"
    if log_file.exists():
        with open(log_file) as f:
            logs = [json.loads(l) for l in f if l.strip()]
        with_loss = [l for l in logs if "loss" in l]
        final_loss = with_loss[-1]["loss"] if with_loss else None
        initial_loss = with_loss[0]["loss"] if with_loss else None
    else:
        final_loss = initial_loss = None

    return {
        "success": True,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "method": method,
        "template": template,
        "duration_seconds": elapsed,
        "output_dir": str(output_dir),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="多模型对比训练")
    parser.add_argument("--data", required=True, help="训练数据 JSON 路径")
    parser.add_argument("--models", default="qwen1.5b",
                       help="逗号分隔模型名: qwen1.5b,qwen7b,llama3.2-3b")
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",")]
    data_path = Path(args.data)
    data = json.load(open(data_path))
    print(f"📊 多模型对比训练")
    print(f"   数据: {data_path} ({len(data)} 条)")
    print(f"   模型: {', '.join(model_keys)}")
    print()

    results = {}
    for key in model_keys:
        if key not in PRESET_MODELS:
            print(f"⚠️  未知模型: {key}")
            continue

        mc = PRESET_MODELS[key]
        out = PROJECT / "data/experiments" / f"benchmark-{key}"

        print(f"━━━ {mc['name']} ({mc['params_b']}B) ━━━")
        model_path = download_model(mc)
        result = train_model(mc, str(data_path), out)
        result["model_name"] = mc["name"]
        result["params_b"] = mc["params_b"]
        results[key] = result
        print()

    # Report
    print("=" * 60)
    print("对比报告")
    print("=" * 60)
    print(f"{'模型':25s} {'Loss':>8s} {'方式':>6s} {'耗时':>10s}")
    print("-" * 55)
    for key, r in results.items():
        if r["success"]:
            loss = f"{r['initial_loss']:.2f}→{r['final_loss']:.2f}"
            dur = f"{r['duration_seconds']/60:.1f}min"
            print(f"{r['model_name']:25s} {loss:>8s} {r['method']:>6s} {dur:>10s}")
        else:
            print(f"{r.get('model_name',key):25s} {'FAILED':>8s}")

    # Save report
    report_path = PROJECT / "data/exports/benchmark_report.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📄 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
