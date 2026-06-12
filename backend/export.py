"""
模型导出：GGUF 格式转换、量化、Model Card 生成

支持将微调后的模型导出为 Ollama / llama.cpp 可用的 GGUF 格式。
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ============================================================
# Configuration
# ============================================================

EXPORTS_DIR = Path(__file__).parent.parent / "data" / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Quantization Options
# ============================================================

QUANTIZATION_OPTIONS = [
    {
        "id": "q4_0",
        "name": "Q4_0",
        "description": "4-bit 量化，最快推理速度",
        "size_ratio": 0.29,  # 相对于 fp16 的大小比例
        "quality": "良好",
        "speed": "最快 ⚡",
        "recommended": False,
    },
    {
        "id": "q4_K_M",
        "name": "Q4_K_M",
        "description": "4-bit 量化，K-quant，平衡质量与速度",
        "size_ratio": 0.34,
        "quality": "优秀 ⭐",
        "speed": "快",
        "recommended": True,
    },
    {
        "id": "q5_K_M",
        "name": "Q5_K_M",
        "description": "5-bit 量化，更好的质量",
        "size_ratio": 0.40,
        "quality": "很好",
        "speed": "较快",
        "recommended": False,
    },
    {
        "id": "q8_0",
        "name": "Q8_0",
        "description": "8-bit 量化，接近原始质量",
        "size_ratio": 0.54,
        "quality": "非常好",
        "speed": "中等",
        "recommended": False,
    },
    {
        "id": "f16",
        "name": "FP16",
        "description": "半精度浮点，无损转换",
        "size_ratio": 1.0,
        "quality": "无损",
        "speed": "慢（需要更多显存）",
        "recommended": False,
    },
]


def get_quantization_options(model_size_gb: float) -> list[dict]:
    """
    获取量化选项列表，包含预估的文件大小。
    """
    options = []
    for opt in QUANTIZATION_OPTIONS:
        option = opt.copy()
        option["estimated_size_gb"] = round(model_size_gb * option["size_ratio"], 2)
        options.append(option)
    return options


# ============================================================
# GGUF Export
# ============================================================

@dataclass
class ExportResult:
    """导出结果"""
    success: bool
    output_path: str = ""
    output_size_gb: float = 0.0
    quantization: str = ""
    duration_seconds: float = 0.0
    error: str = ""


def export_to_gguf(
    model_path: str,
    output_name: str,
    quantization: str = "q4_K_M",
    llama_cpp_path: str | None = None,
    progress_callback=None,
) -> ExportResult:
    """
    将 HuggingFace 格式的模型导出为 GGUF 格式。

    工作流程：
    1. 使用 llama.cpp 的 convert_hf_to_gguf.py 将 HF 模型转为 fp16 GGUF
    2. 使用 llama.cpp 的 quantize 进行量化
    3. 输出最终的 .gguf 文件

    Args:
        model_path: HuggingFace 模型路径
        output_name: 输出文件名（不含扩展名）
        quantization: 量化级别
        llama_cpp_path: llama.cpp 仓库路径
        progress_callback: 进度回调

    Returns:
        ExportResult
    """
    import time
    start_time = time.time()

    output_dir = EXPORTS_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find llama.cpp scripts
    if llama_cpp_path:
        cpp_dir = Path(llama_cpp_path)
    else:
        # Try to find llama-cpp-python's bundled scripts
        try:
            import llama_cpp
            cpp_dir = Path(llama_cpp.__file__).parent.parent / "vendor" / "llama.cpp"
        except ImportError:
            cpp_dir = None

    # Try to use the convert script
    convert_script = None
    possible_paths = [
        cpp_dir / "convert_hf_to_gguf.py" if cpp_dir else None,
        Path("llama.cpp") / "convert_hf_to_gguf.py",
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
    ]

    for p in possible_paths:
        if p and p.exists():
            convert_script = p
            break

    if convert_script is None:
        # Fallback: use llama-cpp-python's built-in conversion
        return _export_via_llama_cpp_python(
            model_path, output_dir, output_name, quantization
        )

    # Step 1: Convert to fp16 GGUF
    fp16_path = output_dir / f"{output_name}.fp16.gguf"

    try:
        cmd = [
            sys.executable,
            str(convert_script),
            "--outfile", str(fp16_path),
            "--outtype", "f16",
            model_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            return ExportResult(
                success=False,
                error=f"GGUF 转换失败:\n{result.stderr[-500:]}",
                quantization=quantization,
            )

    except subprocess.TimeoutExpired:
        return ExportResult(
            success=False,
            error="GGUF 转换超时（超过 1 小时）",
            quantization=quantization,
        )
    except FileNotFoundError:
        return ExportResult(
            success=False,
            error=f"未找到 convert_hf_to_gguf.py。请安装 llama.cpp:\n"
                  f"  git clone https://github.com/ggerganov/llama.cpp.git\n"
                  f"  cd llama.cpp && make",
            quantization=quantization,
        )

    # Step 2: Quantize
    quantize_bin = None
    possible_quantize = [
        cpp_dir / "quantize" if cpp_dir else None,
        Path("llama.cpp") / "quantize",
        Path.home() / "llama.cpp" / "quantize",
    ]
    for p in possible_quantize:
        if p and p.exists():
            quantize_bin = p
            break

    if quantize_bin is None:
        return ExportResult(
            success=False,
            error="未找到 quantize 可执行文件。请先编译 llama.cpp",
            quantization=quantization,
        )

    final_path = output_dir / f"{output_name}.{quantization}.gguf"

    try:
        cmd = [
            str(quantize_bin),
            str(fp16_path),
            str(final_path),
            quantization,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            return ExportResult(
                success=False,
                error=f"量化失败:\n{result.stderr[-500:]}",
                quantization=quantization,
            )

        # Clean up fp16 intermediate
        fp16_path.unlink(missing_ok=True)

        output_size = final_path.stat().st_size / (1024 ** 3)
        elapsed = time.time() - start_time

        return ExportResult(
            success=True,
            output_path=str(final_path),
            output_size_gb=round(output_size, 2),
            quantization=quantization,
            duration_seconds=elapsed,
        )

    except subprocess.TimeoutExpired:
        return ExportResult(
            success=False,
            error="量化超时（超过 1 小时）",
            quantization=quantization,
        )
    except Exception as e:
        return ExportResult(
            success=False,
            error=f"量化过程出错: {str(e)}",
            quantization=quantization,
        )


def _export_via_llama_cpp_python(
    model_path: str,
    output_dir: Path,
    output_name: str,
    quantization: str,
) -> ExportResult:
    """
    使用 llama-cpp-python 库直接导出 GGUF。

    这是备选方案，当 llama.cpp CLI 工具不可用时的兜底方案。
    """
    import time
    start_time = time.time()

    try:
        from llama_cpp import Llama

        # llama-cpp-python can load HF models directly and save as GGUF
        # But this requires the model to be in a supported format

        # Fallback: try to use llama-cpp-python's converter
        try:
            from llama_cpp.llama_cpp import llama_model_quantize
        except ImportError:
            pass

        # The simplest approach: load and save
        # This is limited but works for many models
        final_path = output_dir / f"{output_name}.{quantization}.gguf"

        # Try using the command-line tool that comes with llama-cpp-python
        cmd = [
            sys.executable,
            "-m", "llama_cpp",
            "convert",
            model_path,
            "--outfile", str(final_path),
            "--outtype", quantization if quantization != "f16" else "f16",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0 and final_path.exists():
            output_size = final_path.stat().st_size / (1024 ** 3)
            elapsed = time.time() - start_time
            return ExportResult(
                success=True,
                output_path=str(final_path),
                output_size_gb=round(output_size, 2),
                quantization=quantization,
                duration_seconds=elapsed,
            )

        return ExportResult(
            success=False,
            error=f"llama-cpp-python 导出失败。请手动安装 llama.cpp CLI 工具。\n{result.stderr[-300:]}",
            quantization=quantization,
        )

    except ImportError:
        return ExportResult(
            success=False,
            error="llama-cpp-python 未安装。请运行: pip install llama-cpp-python",
            quantization=quantization,
        )
    except Exception as e:
        return ExportResult(
            success=False,
            error=f"导出异常: {str(e)}",
            quantization=quantization,
        )


# ============================================================
# Model Card Generation
# ============================================================

MODEL_CARD_TEMPLATE = """---
language:
  - {languages}
license: {license}
tags:
{tags_block}
pipeline_tag: text-generation
base_model: {base_model}
---

# {model_name}

{description}

## 模型信息

- **基础模型**: {base_model}
- **微调方式**: {finetuning_type}
- **训练数据**: {dataset_name} ({dataset_size} 条)
- **训练参数**:
  - Learning Rate: {learning_rate}
  - Epochs: {num_epochs}
  - LoRA Rank: {lora_rank}
  - Batch Size: {batch_size}

## 使用方法

### Ollama

首先创建 Modelfile:

```dockerfile
FROM {gguf_filename}

TEMPLATE \"\"\"{chat_template}\"\"\"
```

然后运行:

```bash
ollama create {ollama_name}
ollama run {ollama_name}
```

### llama.cpp

```bash
./llama-cli -m {gguf_filename} -p "你好"
```

## 训练信息

- **训练平台**: 模型训练室 (Model Training Room)
- **训练耗时**: {training_duration}
- **最终 Loss**: {final_loss}
- **导出格式**: GGUF ({quantization})

## 许可证

{license_info}

## 免责声明

本模型由 {creator} 使用模型训练室软件微调生成。
请遵守基础模型的许可证要求。
"""


def generate_model_card(
    model_name: str,
    base_model: str,
    finetuning_type: str = "lora",
    dataset_name: str = "",
    dataset_size: int = 0,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    lora_rank: int = 16,
    batch_size: int = 4,
    gguf_filename: str = "",
    ollama_name: str = "",
    quantization: str = "q4_K_M",
    chat_template: str = "",
    final_loss: float = 0.0,
    training_duration: str = "",
    license_info: str = "请遵守基础模型的原始许可证",
    creator: str = "模型训练室用户",
    description: str = "",
    languages: str = "zh",
    tags: list[str] | None = None,
) -> str:
    """
    生成 HuggingFace 标准的 Model Card。
    """
    if tags is None:
        tags = ["fine-tuned", "gguf"]

    tags_block = "\n".join(f"  - {tag}" for tag in tags)

    if not description:
        description = f"{model_name} 是基于 {base_model} 微调的模型。"

    return MODEL_CARD_TEMPLATE.format(
        model_name=model_name,
        base_model=base_model,
        description=description,
        finetuning_type=finetuning_type.upper(),
        dataset_name=dataset_name or "自定义数据集",
        dataset_size=f"{dataset_size:,}",
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        lora_rank=lora_rank,
        batch_size=batch_size,
        gguf_filename=gguf_filename,
        ollama_name=ollama_name,
        quantization=quantization,
        chat_template=chat_template,
        final_loss=f"{final_loss:.4f}",
        training_duration=training_duration or "未知",
        license_info=license_info,
        creator=creator,
        languages=languages,
        license=license_info,
        tags_block=tags_block,
    )


# ============================================================
# Ollama Integration
# ============================================================

def generate_ollama_modelfile(
    gguf_path: str,
    model_name: str,
    chat_template: str = "",
    system_prompt: str = "",
    temperature: float = 0.7,
) -> str:
    """
    生成 Ollama Modelfile 内容。
    """
    lines = [
        f"FROM {gguf_path}",
        "",
    ]

    if chat_template:
        lines.append(f'TEMPLATE """{chat_template}"""')
        lines.append("")

    if system_prompt:
        lines.append(f'SYSTEM """{system_prompt}"""')
        lines.append("")

    lines.append(f"PARAMETER temperature {temperature}")
    lines.append("PARAMETER top_p 0.9")
    lines.append("PARAMETER top_k 40")

    return "\n".join(lines)


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📦 量化选项")
    print("=" * 60)
    options = get_quantization_options(14.5)  # 7B model
    for opt in options:
        rec = " ⭐ 推荐" if opt["recommended"] else ""
        print(f"  {opt['name']}: {opt['description']}")
        print(f"    预估大小: {opt['estimated_size_gb']:.1f} GB · 质量: {opt['quality']}{rec}")

    print()
    print("=" * 60)
    print("📋 Model Card 示例")
    print("=" * 60)
    card = generate_model_card(
        model_name="My-Chinese-Assistant",
        base_model="Qwen2.5-7B-Instruct",
        dataset_name="Chinese-Alpaca-50K",
        dataset_size=50000,
        gguf_filename="my-chinese-assistant.Q4_K_M.gguf",
        ollama_name="my-assistant",
        final_loss=1.24,
        training_duration="3 小时 12 分",
    )
    print(card[:500] + "...")

    print()
    print("=" * 60)
    print("🤖 Ollama Modelfile 示例")
    print("=" * 60)
    modelfile = generate_ollama_modelfile(
        gguf_path="./my-assistant.Q4_K_M.gguf",
        model_name="my-assistant",
        chat_template="<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n",
        system_prompt="你是一个乐于助人的中文AI助手。",
    )
    print(modelfile)

    print()
    print("✅ 导出模块自检完成")
