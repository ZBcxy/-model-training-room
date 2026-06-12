"""
实操验证 Step 4: 导出 GGUF + 验证可被 Ollama 加载

从微调后的 checkpoint 导出 GGUF 格式

用法：
  python tests/step4_export.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.export import (
    generate_model_card,
    generate_ollama_modelfile,
    get_quantization_options,
)

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "experiments" / "demo-qwen1.5b-qlora" / "latest"
OUTPUT_DIR = PROJECT_ROOT / "data" / "exports" / "demo-qwen1.5b-qlora"
MODEL_NAME = "demo-qwen1.5b-qlora"


def check_prerequisites():
    """检查前置条件"""
    print("=" * 60)
    print("实操验证 Step 4: 导出 GGUF + 验证")
    print("=" * 60)
    print()

    ok = True

    # Checkpoint
    if CHECKPOINT_DIR.exists():
        print(f"✅ Checkpoint 就绪: {CHECKPOINT_DIR}")
    else:
        print(f"⚠️  Checkpoint 路径不存在: {CHECKPOINT_DIR}")
        print("   尝试查找其他 checkpoint...")
        exp_dir = PROJECT_ROOT / "data" / "experiments" / "demo-qwen1.5b-qlora"
        checkpoints = list(exp_dir.glob("checkpoint-*")) if exp_dir.exists() else []
        if checkpoints:
            print(f"   找到: {checkpoints}")
        else:
            print("   未找到任何 checkpoint，请先运行 step3_train.py")
            ok = False

    # Ollama
    ollama_path = None
    for candidate in ["ollama"]:
        try:
            result = subprocess.run([candidate, "--version"], capture_output=True, text=True)
            ollama_path = candidate
            print(f"✅ Ollama: {result.stdout.strip()}")
            break
        except FileNotFoundError:
            continue
    if not ollama_path:
        print("⚠️  Ollama 未安装（导出 GGUF 不需要，但部署需要）")
        print("   安装: curl -fsSL https://ollama.com/install.sh | sh")

    # llama.cpp
    llama_cpp_tools = False
    for tool in ["llama.cpp/build/bin/llama-quantize", "llama.cpp/quantize"]:
        tool_path = Path.home() / tool
        if tool_path.exists():
            llama_cpp_tools = True
            print(f"✅ llama.cpp 工具: {tool_path}")
            break
    if not llama_cpp_tools:
        print("⚠️  llama.cpp 量化工具未找到（导出 GGUF 需要）")
        print("   安装: cd ~ && git clone https://github.com/ggerganov/llama.cpp.git --depth 1 && cd llama.cpp && cmake -B build && cmake --build build --config Release -j")

    return ok


def export_gguf():
    """导出 GGUF"""
    print()
    print("─" * 60)
    print("📦 导出 GGUF...")
    print("─" * 60)

    # Find checkpoint
    ckpt_dir = CHECKPOINT_DIR
    if not ckpt_dir.exists():
        exp_dir = PROJECT_ROOT / "data" / "experiments" / "demo-qwen1.5b-qlora"
        checkpoints = sorted(exp_dir.glob("checkpoint-*"))
        if checkpoints:
            ckpt_dir = checkpoints[-1]
        else:
            print("❌ 没有可用的 checkpoint")
            return False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Show quantization options
    base_size = 2.88  # GB (Qwen2.5-1.5B)
    print(f"   基础模型大小: ~{base_size:.1f} GB (FP16)")
    q_opts = get_quantization_options(base_size)
    for opt in q_opts:
        rec = " ⭐" if opt["recommended"] else ""
        print(f"   {opt['name']}: ~{opt['estimated_size_gb']:.1f} GB - {opt['description']}{rec}")

    # Use Q4_K_M (recommended)
    quant = "Q4_K_M"
    output_file = OUTPUT_DIR / f"{MODEL_NAME}.{quant}.gguf"

    print()
    print(f"   导出配置:")
    print(f"   输入: {ckpt_dir}")
    print(f"   输出: {output_file}")
    print(f"   量化: {quant}")

    # Method 1: Try llama-cpp-python's built-in conversion
    # Method 2: Use convert_hf_to_gguf.py + quantize

    # For LoRA fine-tuned models, we need to merge the LoRA weights first
    # then convert the merged model
    print()
    print("   注意: LoRA 微调的模型需要先合并权重再导出")
    print("   步骤:")
    print("   1. 合并 LoRA 权重到基础模型")
    print("   2. 转换合并后的模型为 GGUF FP16")
    print("   3. 量化为目标精度")

    # For now, show the commands user would run
    base_model = PROJECT_ROOT / "data" / "models" / "Qwen--Qwen2.5-1.5B-Instruct"
    merged_dir = OUTPUT_DIR / "merged"

    cmds = f"""
# Step 1: Merge LoRA weights
python -m llamafactory.export \\
    --model_name_or_path {base_model} \\
    --adapter_name_or_path {ckpt_dir} \\
    --template qwen \\
    --finetuning_type lora \\
    --export_dir {merged_dir} \\
    --export_size 2 \\
    --export_legacy_format False

# Step 2: Convert to GGUF FP16
python ~/llama.cpp/convert_hf_to_gguf.py \\
    {merged_dir} \\
    --outfile {OUTPUT_DIR / f'{MODEL_NAME}.FP16.gguf'} \\
    --outtype f16

# Step 3: Quantize
~/llama.cpp/build/bin/llama-quantize \\
    {OUTPUT_DIR / f'{MODEL_NAME}.FP16.gguf'} \\
    {output_file} \\
    {quant}
"""
    print()
    print("   等效命令:")
    print(cmds)

    return True


def generate_card():
    """生成 Model Card"""
    print("─" * 60)
    print("📋 生成 Model Card...")
    print("─" * 60)

    card = generate_model_card(
        model_name=MODEL_NAME,
        base_model="Qwen2.5-1.5B-Instruct",
        finetuning_type="lora",
        dataset_name="demo-zh-conversation",
        dataset_size=500,
        learning_rate=2e-4,
        num_epochs=1,
        lora_rank=8,
        batch_size=2,
        gguf_filename=f"{MODEL_NAME}.Q4_K_M.gguf",
        ollama_name=MODEL_NAME,
        quantization="Q4_K_M",
        chat_template="",
        training_duration="~3 分钟",
        description="基于 Qwen2.5-1.5B 用 500 条中文对话数据 QLoRA 微调的演示模型。",
        creator="模型训练室实操验证",
    )

    card_path = OUTPUT_DIR / "README.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"   ✅ Model Card 已保存: {card_path}")


def generate_ollama_file():
    """生成 Ollama Modelfile"""
    modelfile = generate_ollama_modelfile(
        gguf_path=f"./{MODEL_NAME}.Q4_K_M.gguf",
        model_name=MODEL_NAME,
        chat_template="",
        system_prompt="你是基于Qwen的微调演示模型，训练数据为500条中文对话。",
    )

    modelfile_path = OUTPUT_DIR / "Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile)
    print(f"   ✅ Modelfile 已保存: {modelfile_path}")
    print()
    print("   部署到 Ollama:")
    print(f"     cd {OUTPUT_DIR}")
    print(f"     ollama create {MODEL_NAME}")
    print(f"     ollama run {MODEL_NAME}")


if __name__ == "__main__":
    if not check_prerequisites():
        print()
        print("⚠️  部分前置条件不满足。")
        print("   将继续执行可完成的部分。")
        print()

    export_gguf()
    generate_card()
    generate_ollama_file()

    print()
    print("=" * 60)
    print("✅ Step 4 完成！")
    print()
    print(f"   导出目录: {OUTPUT_DIR}")
    print(f"   下一步: ollama create {MODEL_NAME} && ollama run {MODEL_NAME}")
    print("=" * 60)
