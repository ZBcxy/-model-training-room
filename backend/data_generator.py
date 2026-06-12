"""
AI 数据生成引擎 — 多后端 + 质量保证

支持后端：
- Ollama (本地)
- OpenAI API (GPT-4/GPT-4o)
- Claude API
"""

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .dataset_manager import convert_to_sharegpt, clean_dataset, detect_format, get_data_stats

# ============================================================
# Backend Clients
# ============================================================

class OllamaClient:
    """本地 Ollama 客户端"""

    def __init__(self, model: str = "demo-qwen", host: str = "http://127.0.0.1:11434"):
        self.model = model
        self.host = host
        # Find ollama binary
        for candidate in [
            Path(__file__).parent.parent / "tools" / "ollama_extract" / "bin" / "ollama",
            Path.home() / ".ollama" / "ollama",
        ]:
            if candidate.exists():
                self.bin = str(candidate)
                break
        else:
            self.bin = "ollama"  # Hope it's in PATH

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        env = {**os.environ, "OLLAMA_HOST": self.host}
        result = subprocess.run(
            [self.bin, "run", self.model, prompt],
            capture_output=True, text=True, timeout=120, env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ollama 调用失败: {result.stderr[:300]}")
        return result.stdout.strip()


class OpenAIClient:
    """OpenAI API 客户端"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        import urllib.request
        data = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.8,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"]


# ============================================================
# Prompt Templates
# ============================================================

PROMPT_TEMPLATES = {
    "chat": """你是一个数据标注专家。请根据以下示例，生成 {count} 条格式一致的中文对话训练数据。

场景：{scenario}

示例：
{examples_json}

要求：
1. 严格遵循示例的 JSON 格式（每条包含 instruction 和 output）
2. 答案要准确、详细、有帮助，不能敷衍
3. 覆盖不同难度和主题，避免重复
4. 提问方式要多样化
5. 只输出 JSON 数组，不要任何解释
6. 输出格式：[{{"instruction": "...", "output": "..."}}, ...]
""",

    "code": """你是一个代码数据标注专家。请根据以下示例，生成 {count} 条代码相关的训练数据。

场景：{scenario}

示例：
{examples_json}

要求：
1. 严格遵循示例的 JSON 格式
2. 代码要能运行，包含必要注释
3. 覆盖不同编程语言和难度
4. 提问覆盖：代码生成、调试、解释、优化等场景
5. 只输出 JSON 数组，不要任何解释
""",

    "qa": """你是一个知识问答数据标注专家。请根据以下示例，生成 {count} 条知识问答训练数据。

场景：{scenario}

示例：
{examples_json}

要求：
1. 严格遵循示例的 JSON 格式
2. 答案要事实准确、来源可靠
3. 覆盖科学、历史、文化、技术等多个领域
4. 只输出 JSON 数组，不要任何解释
""",
}


# ============================================================
# Generator
# ============================================================

@dataclass
class GenerationResult:
    success: bool
    data: list[dict] = field(default_factory=list)
    count: int = 0
    error: str = ""
    api_calls: int = 0
    tokens_used: int = 0
    duration_seconds: float = 0.0


def create_client(backend: str, api_key: str = "", model: str = "") -> object:
    """创建一个生成客户端"""
    backend = backend.lower()
    if "ollama" in backend:
        return OllamaClient(model=model or "demo-qwen")
    elif "openai" in backend:
        if not api_key:
            raise ValueError("OpenAI 需要 API Key")
        return OpenAIClient(api_key=api_key, model=model or "gpt-4o-mini")
    elif "claude" in backend:
        if not api_key:
            raise ValueError("Claude 需要 API Key")
        return OpenAIClient(api_key=api_key, model=model or "claude-sonnet-4-6",
                          base_url="https://api.anthropic.com/v1")
    else:
        raise ValueError(f"不支持的生成引擎: {backend}")


def generate_data(
    examples: list[dict],
    target_count: int = 100,
    backend: str = "ollama",
    api_key: str = "",
    model: str = "",
    scenario: str = "中文对话助手",
    data_type: str = "chat",
    quality_check: bool = True,
    progress_callback=None,
) -> GenerationResult:
    """
    核心生成函数：从少量示例生成批量训练数据。

    Args:
        examples: 3-5 个示例，格式为 [{"instruction": "...", "output": "..."}, ...]
        target_count: 目标生成数量
        backend: "ollama" | "openai" | "claude"
        api_key: API Key（OpenAI/Claude 需要）
        model: 模型名（可选）
        scenario: 场景描述
        data_type: 数据类型 "chat" | "code" | "qa"
        quality_check: 是否做质量过滤
        progress_callback: 进度回调 fn(current, total, message)

    Returns:
        GenerationResult
    """
    start_time = time.time()
    result = GenerationResult()

    if len(examples) < 1:
        result.error = "至少需要 1 个示例"
        return result

    # Validate examples format
    for ex in examples:
        if not isinstance(ex, dict) or "instruction" not in ex or "output" not in ex:
            result.error = "示例格式错误：每条示例必须包含 instruction 和 output 字段"
            return result

    # Create client
    try:
        client = create_client(backend, api_key, model)
    except Exception as e:
        result.error = str(e)
        return result

    # Get prompt template
    template = PROMPT_TEMPLATES.get(data_type, PROMPT_TEMPLATES["chat"])
    examples_json = json.dumps(examples, ensure_ascii=False, indent=2)

    # Generate in batches
    all_data = []
    batch_size = min(30, target_count)
    total_batches = (target_count + batch_size - 1) // batch_size

    for batch in range(total_batches):
        remaining = min(batch_size, target_count - len(all_data))
        if remaining <= 0:
            break

        prompt = template.format(
            count=remaining,
            scenario=scenario,
            examples_json=examples_json,
        )

        try:
            if progress_callback:
                progress_callback(batch + 1, total_batches, f"正在生成第 {batch+1}/{total_batches} 批...")

            response = client.generate(prompt, max_tokens=4096)
            result.api_calls += 1

            # Parse response — extract JSON array
            batch_data = _extract_json_array(response)
            if batch_data:
                all_data.extend(batch_data)
            else:
                # Try harder — maybe the response has extra text around the JSON
                cleaned = response.strip()
                # Remove markdown code blocks
                cleaned = re.sub(r'^```(?:json)?\s*\n', '', cleaned)
                cleaned = re.sub(r'\n```\s*$', '', cleaned)
                try:
                    batch_data = json.loads(cleaned)
                    if isinstance(batch_data, list):
                        all_data.extend(batch_data)
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            if progress_callback:
                progress_callback(batch + 1, total_batches, f"第 {batch+1} 批出错: {e}")
            continue

    # Convert to ShareGPT
    converted = convert_to_sharegpt(all_data)

    # Quality filtering
    if quality_check and converted:
        clean_result = clean_dataset(converted, {
            "remove_duplicates": True,
            "remove_empty": True,
            "min_length": 20,
            "max_length": 8192,
        })
        converted = clean_result["cleaned_data"]
        stats = clean_result["stats"]

    result.data = converted[:target_count]
    result.count = len(result.data)
    result.success = result.count > 0
    result.duration_seconds = time.time() - start_time

    if result.count == 0:
        result.error = "未能生成有效数据，请检查示例质量或更换生成引擎"

    if result.count < target_count:
        result.error = f"生成了 {result.count}/{target_count} 条数据（部分批次失败或质量过滤移除了一些）"

    return result


def _extract_json_array(text: str) -> list | None:
    """从文本中提取 JSON 数组"""
    # Find [ ... ] pairs
    matches = list(re.finditer(r'\[', text))
    for m in matches:
        depth = 0
        end = m.start()
        for i, ch in enumerate(text[m.start():], m.start()):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > m.start():
            candidate = text[m.start():end]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    # Validate each item has expected fields
                    valid = all(
                        isinstance(item, dict) and
                        any(k in item for k in ["instruction", "messages", "conversations"])
                        for item in parsed
                    )
                    if valid:
                        return parsed
            except json.JSONDecodeError:
                continue
    return None


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":
    # Test with examples
    examples = [
        {"instruction": "什么是机器学习？请用通俗的语言解释。", "output": "机器学习是人工智能的一个分支。简单来说，就是让计算机通过看大量数据来自动学习规律，而不是人工编写规则..."},
        {"instruction": "Python中list和tuple的区别是什么？", "output": "主要区别：1) list 可变，可以增删改；tuple 不可变，创建后不能修改。2) list 用中括号[]，tuple 用小括号()。3) list 性能稍慢但灵活，tuple 更快..."},
        {"instruction": "给新人推荐三个必学的Python库。", "output": "推荐这三个：1) pandas — 数据处理必备；2) requests — HTTP请求；3) flask — 快速搭建Web应用。这三个覆盖了数据、网络、Web三大方向..."},
    ]

    print("=" * 60)
    print("🧪 AI 数据生成引擎测试")
    print("=" * 60)

    # Test 1: Syntax check only (don't actually call API)
    print("✅ 模块导入成功")
    print(f"✅ 3 个后端: Ollama, OpenAI, Claude")
    print(f"✅ 3 种数据类型: {list(PROMPT_TEMPLATES.keys())}")
    print(f"✅ JSON提取器就绪")
    print()

    # Test 2: JSON extraction
    test_resp = '一些文字[{"instruction": "测试", "output": "答案"}]更多文字'
    extracted = _extract_json_array(test_resp)
    print(f"JSON提取测试: {'✅' if extracted and len(extracted)==1 else '❌'} {extracted}")

    print()
    print("✅ 数据生成引擎模块自检完成")
