"""
实操验证 Step 1: 下载真实模型

使用 ModelScope SDK 下载 Qwen2.5-1.5B-Instruct (~3GB)
验证国内网络下完整下载流程
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hardware_checker import get_system_info


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_DIR = Path(__file__).parent.parent / "data" / "models" / "Qwen--Qwen2.5-1.5B-Instruct"


def download_via_modelscope():
    """通过 ModelScope 下载（国内更快）"""
    from modelscope import snapshot_download

    print(f"📥 通过 ModelScope 下载: {MODEL_ID}")
    print(f"   目标目录: {MODEL_DIR}")

    start = time.time()

    try:
        local_path = snapshot_download(
            model_id=MODEL_ID,
            cache_dir=str(MODEL_DIR.parent),
            local_dir=str(MODEL_DIR),
        )
        elapsed = time.time() - start
        print(f"   ✅ 下载完成！")
        print(f"   路径: {local_path}")
        print(f"   耗时: {elapsed:.0f} 秒")

        # Check size
        total_size = sum(
            f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file()
        )
        print(f"   大小: {total_size / (1024**3):.2f} GB")

        return True

    except Exception as e:
        print(f"   ❌ ModelScope 下载失败: {e}")
        return False


def download_via_huggingface():
    """通过 HuggingFace 下载（备选）"""
    from huggingface_hub import snapshot_download as hf_snapshot

    print(f"📥 通过 HuggingFace 下载: {MODEL_ID}")

    start = time.time()

    try:
        # Try with HF mirror
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        local_path = hf_snapshot(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_DIR),
            resume_download=True,
            max_workers=4,
        )
        elapsed = time.time() - start
        print(f"   ✅ 下载完成！")
        print(f"   路径: {local_path}")
        print(f"   耗时: {elapsed:.0f} 秒")

        total_size = sum(
            f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file()
        )
        print(f"   大小: {total_size / (1024**3):.2f} GB")

        return True

    except Exception as e:
        print(f"   ❌ HuggingFace 下载失败: {e}")
        return False


def verify_model():
    """验证下载的模型是否可用"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print()
    print("🔍 验证模型...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR),
            trust_remote_code=True,
        )
        print(f"   ✅ Tokenizer 加载成功 (vocab: {tokenizer.vocab_size})")

        # Quick test: just load config, don't load full weights
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        print(f"   ✅ Config 加载成功")
        print(f"   架构: {config.model_type}")
        print(f"   隐藏层维度: {config.hidden_size}")
        print(f"   层数: {config.num_hidden_layers}")

        # Test tokenizer
        test_text = "你好，请问今天天气怎么样？"
        tokens = tokenizer.encode(test_text)
        print(f"   测试 Tokenize: '{test_text}' → {len(tokens)} tokens")

        return True

    except Exception as e:
        print(f"   ❌ 验证失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("实操验证 Step 1: 下载真实模型")
    print("=" * 60)

    # Show system info
    info = get_system_info()
    print(f"GPU: {info.gpus[0].name if info.has_gpu else 'N/A'}")
    print(f"磁盘可用: {info.disk_free_gb:.1f} GB")
    print(f"模型大小预估: ~3 GB")
    print()

    # Check if already downloaded
    if MODEL_DIR.exists() and any(MODEL_DIR.glob("*.safetensors")):
        print("⚠️  模型已存在，跳过下载")
        verify_model()
        sys.exit(0)

    # Try ModelScope first, then HF
    success = download_via_modelscope()
    if not success:
        print()
        print("🔄 尝试 HuggingFace 镜像...")
        success = download_via_huggingface()

    if success:
        verify_model()
    else:
        print()
        print("💡 手动下载提示:")
        print(f"   1. 访问 https://modelscope.cn/models/{MODEL_ID}")
        print("   2. 下载所有文件")
        print(f"   3. 放到 {MODEL_DIR}")
        sys.exit(1)
