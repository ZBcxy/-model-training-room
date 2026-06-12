#!/usr/bin/env python3
"""
多模型对比训练脚本

用同一份数据训练多个模型，自动生成对比报告。

用法:
  python tools/compare_models.py --data data/datasets/demo-final/data.json --models qwen1.5b,llama3.2-3b
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

# ── 对比模型池 ──────────────────────────────────────────
POOL = {
    "qwen1.5b": {
        "name": "Qwen2.5-1.5B", "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "source": "modelscope", "params_b": 1.5, "template": "qwen",
    },
    "qwen7b": {
        "name": "Qwen2.5-7B", "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "source": "modelscope", "params_b": 7.0, "template": "qwen",
    },
    "llama3.2-3b": {
        "name": "Llama-3.2-3B", "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "source": "huggingface", "params_b": 3.0, "template": "llama3",
    },
    "chatglm3-6b": {
        "name": "ChatGLM3-6B", "model_id": "ZhipuAI/chatglm3-6b",
        "source": "modelscope", "params_b": 6.0, "template": "chatglm3",
    },
}


def download(model_key):
    """下载模型"""
    info = POOL[model_key]
    target = PROJECT / "data/models" / info["model_id"].replace("/", "--")
    if (target / "model.safetensors").exists():
        print(f"  ✅ 已缓存")
        return target

    print(f"  📥 下载中...")
    from modelscope import snapshot_download
    return Path(snapshot_download(info["model_id"], cache_dir=str(target)))


def train(model_key, data_path):
    """训练模型"""
    import yaml
    import subprocess

    info = POOL[model_key]
    model_dir = download(model_key)
    out = PROJECT / "data/experiments" / f"compare-{model_key}"

    if (out / "adapter_model.safetensors").exists():
        print(f"  ✅ 已训练，跳过")
        return _read_metrics(out)

    out.mkdir(parents=True, exist_ok=True)

    # 自动判断用 LoRA 还是 QLoRA
    from backend.hardware_checker import get_system_info
    hw = get_system_info()
    vram = hw.max_single_vram_gb if hw.has_gpu else 8.0
    method = "lora" if vram >= 8 else "qlora"
    lora_r = 16 if info["params_b"] < 10 else 32

    config = {
        "model_name_or_path": str(model_dir),
        "finetuning_type": method,
        "template": info["template"],
        "dataset": "compare_data",
        "dataset_dir": str(Path(data_path).parent),
        "cutoff_len": 2048, "output_dir": str(out),
        "logging_steps": 10, "save_steps": 200, "plot_loss": True,
        "overwrite_output_dir": True, "do_train": True,
        "lora_rank": lora_r, "lora_alpha": lora_r * 2, "lora_dropout": 0.05,
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4, "num_train_epochs": 2,
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
        "optim": "adamw_torch", "gradient_checkpointing": True,
        "report_to": "none", "bf16": True,
    }

    yaml_path = out / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, allow_unicode=True)

    print(f"  🚀 训练: {info['name']} ({method.upper()})")
    start = time.time()
    r = subprocess.run([
        sys.executable, str(PROJECT / "LLaMA-Factory/src/train.py"), str(yaml_path)
    ])
    elapsed = time.time() - start

    if r.returncode != 0:
        return {"error": f"Exit code {r.returncode}", "duration_seconds": elapsed}

    return _read_metrics(out)


def _read_metrics(out_dir):
    """从训练日志读取指标"""
    log_file = out_dir / "trainer_log.jsonl"
    if not log_file.exists():
        return {"error": "No log file"}

    with open(log_file) as f:
        logs = [json.loads(l) for l in f if l.strip()]
    with_loss = [l for l in logs if "loss" in l]
    last = logs[-1]

    return {
        "initial_loss": with_loss[0]["loss"] if with_loss else None,
        "final_loss": with_loss[-1]["loss"] if with_loss else None,
        "duration": last.get("elapsed_time", "?"),
        "steps": last.get("current_steps", 0),
        "epochs": last.get("epoch", 0),
    }


def evaluate(model_key):
    """推理评测"""
    info = POOL[model_key]
    ollama_model = f"compare-{model_key}"

    # 合并 + 导出 GGUF
    merged = PROJECT / "data/exports" / f"compare-{model_key}" / "merged"
    if not (merged / "model-00001-of-00002.safetensors").exists():
        print(f"  🔀 合并权重...")
        import subprocess
        ckpt = PROJECT / "data/experiments" / f"compare-{model_key}"
        r = subprocess.run([
            sys.executable, "-m", "llamafactory.cli", "export",
            "--model_name_or_path",
            str(PROJECT / "data/models" / info["model_id"].replace("/", "--")),
            "--adapter_name_or_path", str(ckpt),
            "--template", info["template"],
            "--finetuning_type", "lora",
            "--export_dir", str(merged),
            "--export_size", "2",
        ], capture_output=True)
        if r.returncode != 0:
            return {"error": f"Merge failed: {r.stderr[-200:]}"}

    # GGUF + Ollama
    # (需要 llama.cpp 和 ollama，如果不存在则跳过)
    return {"merged": str(merged), "status": "ready for GGUF export"}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--models", default="qwen1.5b")
    p.add_argument("--evaluate", action="store_true")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    data = json.load(open(args.data))
    print(f"📊 多模型对比")
    print(f"   数据: {args.data} ({len(data)} 条)")
    print(f"   模型: {', '.join(m[0] for m in [POOL[k]['name'] for k in models])}")
    print()

    results = {}
    for key in models:
        if key not in POOL:
            print(f"⚠️  未知: {key}")
            continue
        print(f"━━━ {POOL[key]['name']} ({POOL[key]['params_b']}B) ━━━")
        try:
            r = train(key, args.data)
            r["model"] = POOL[key]["name"]
            r["params_b"] = POOL[key]["params_b"]
            results[key] = r
            if r.get("final_loss"):
                print(f"  ✅ Loss: {r['initial_loss']:.2f}→{r['final_loss']:.2f} ({r['duration']})")
        except Exception as e:
            results[key] = {"error": str(e), "model": POOL[key]["name"]}
            print(f"  ❌ {e}")
        print()

    # 对比报告
    print("=" * 60)
    print("对比报告")
    print("=" * 60)
    print(f"{'模型':25s} {'参数量':>6s} {'Loss':>12s} {'耗时':>10s}")
    print("-" * 55)
    for key, r in results.items():
        if r.get("final_loss"):
            loss = f"{r['initial_loss']:.2f}→{r['final_loss']:.2f}"
            print(f"{r['model']:25s} {r['params_b']:4.1f}B {loss:>12s} {r['duration']:>10s}")
        else:
            print(f"{r.get('model',key):25s} {'—':>6s} {'FAILED':>12s}")

    # 保存
    out = PROJECT / "data/exports/compare_report.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📄 {out}")


if __name__ == "__main__":
    main()
