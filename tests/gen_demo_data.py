"""
生成 Demo 训练数据 —— 用 Ollama 批量生成高质量中文对话
4 个主题 × 250条 = 1000 条
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "data/datasets/demo-1k-zh"
OLLAMA_BIN = str(PROJECT_ROOT / "tools/ollama_extract/bin/ollama")

TOPICS = {
    "日常对话": "日常中文对话，包括问候、聊天、建议、情感交流",
    "知识问答": "中文知识问答，覆盖科学、历史、文化、地理等领域",
    "写作助手": "中文写作帮助，包括文案、邮件、诗歌、故事等创作",
    "实用技能": "实用技能问答，包括烹饪、健康、旅游、学习方法等",
}


def generate(topic_name, topic_desc, count, batch_size=30):
    """生成一批数据"""
    data = []
    prompt = f"""你是数据标注专家。请生成{batch_size}条「{topic_name}」类的训练数据。

场景：{topic_desc}

格式：严格JSON数组，每条包含 instruction 和 output 字段。
要求：答案准确详细，有实际帮助价值。只输出JSON，不解释。
示例格式：[{{"instruction":"问题","output":"回答"}}]
"""
    remaining = count
    while remaining > 0:
        n = min(batch_size, remaining)
        p = prompt.replace(f"生成{batch_size}条", f"生成{n}条")
        try:
            r = subprocess.run(
                [OLLAMA_BIN, "run", "demo-qwen", p],
                capture_output=True, text=True, timeout=180,
                env={**os.environ, "OLLAMA_HOST": "http://127.0.0.1:11434"},
            )
            resp = r.stdout.strip()
            # Extract JSON
            match = re.search(r'\[.*\]', resp, re.DOTALL)
            if match:
                batch = json.loads(match.group())
                valid = [item for item in batch if isinstance(item, dict) and "instruction" in item and "output" in item and len(item.get("output", "")) > 20]
                data.extend(valid)
                remaining -= len(valid)
                print(f"  {topic_name}: {len(data)}/{count}")
            else:
                remaining -= n  # skip this batch
        except Exception as e:
            print(f"  Error: {e}")
            remaining -= n
        time.sleep(1)  # rate limit
    return data


def main():
    print("=" * 60)
    print("生成 1000 条中文对话训练数据")
    print("=" * 60)

    all_data = []
    for topic, desc in TOPICS.items():
        print(f"\n{topic}...")
        data = generate(topic, desc, 250)
        all_data.extend(data)
        print(f"  → {len(data)} 条")

    # Dedup by instruction
    seen = set()
    unique = []
    for item in all_data:
        key = item["instruction"][:50]
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # Limit to 1000
    unique = unique[:1000]

    # Save
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT / "data.json", "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    with open(OUTPUT / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "name": "demo-1k-zh",
            "record_count": len(unique),
            "topics": list(TOPICS.keys()),
            "format": "sharegpt",
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 保存 {len(unique)} 条 → {OUTPUT / 'data.json'}")

    # Preview
    print("\n预览:")
    for i, item in enumerate(unique[:3]):
        inst = item["instruction"][:80]
        out = item["output"][:80]
        print(f"  [{i+1}] Q: {inst}...")
        print(f"      A: {out}...")


if __name__ == "__main__":
    main()
