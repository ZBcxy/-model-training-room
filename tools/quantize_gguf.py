#!/usr/bin/env python
"""Quantize GGUF model using llama-cpp-python's built-in quantizer."""
import sys
import ctypes
from pathlib import Path
from llama_cpp.llama_cpp import (
    llama_model_quantize,
    llama_model_quantize_params,
    llama_model_quantize_default_params,
)

# Quantization type mapping
FTYPE_MAP = {
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 6,
    "q5_1": 7,
    "q8_0": 8,
    "q4_K_M": 15,
    "q5_K_M": 17,
    "q6_K": 18,
    "q8_K": 19,
}

def quantize(input_path: str, output_path: str, qtype: str = "q4_K_M"):
    """Quantize a GGUF FP16 model to a smaller format."""
    inp = Path(input_path)
    out = Path(output_path)

    if not inp.exists():
        print(f"❌ Input file not found: {inp}")
        return False

    # Setup params
    params = llama_model_quantize_default_params()
    params.ftype = FTYPE_MAP.get(qtype, 15)

    print(f"📦 Quantizing: {inp.name}")
    print(f"   Input:   {inp.stat().st_size / (1024**3):.1f} GB")
    print(f"   Output:  {out}")
    print(f"   Type:    {qtype} (ftype={params.ftype})")
    print(f"   Threads: {params.nthread}")

    # Quantize
    inp_bytes = str(inp).encode('utf-8')
    out_bytes = str(out).encode('utf-8')

    result = llama_model_quantize(inp_bytes, out_bytes, ctypes.byref(params))

    if result == 0:
        out_size = out.stat().st_size / (1024**3)
        ratio = out_size / (inp.stat().st_size / (1024**3)) * 100
        print(f"   ✅ Done! Output: {out_size:.2f} GB ({ratio:.0f}% of original)")
        return True
    else:
        print(f"   ❌ Quantization failed (error code: {result})")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantize GGUF model")
    parser.add_argument("input", help="Input FP16 GGUF file")
    parser.add_argument("output", help="Output quantized GGUF file")
    parser.add_argument("--type", default="q4_K_M", choices=list(FTYPE_MAP.keys()),
                        help="Quantization type (default: q4_K_M)")
    args = parser.parse_args()

    success = quantize(args.input, args.output, args.type)
    sys.exit(0 if success else 1)
