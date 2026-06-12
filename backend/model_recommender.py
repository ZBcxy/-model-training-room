"""
智能模型推荐引擎 — 硬件 + 语言 + 任务三维匹配

超越简单的 pattern 匹配，综合考虑：
- 硬件约束（VRAM、RAM）
- 语言偏好（中文、英文、双语）
- 任务类型（对话、代码、翻译、安全对齐）
- 用户级别（入门、进阶、专业）
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .hardware_checker import get_system_info, recommend_method
from .model_cards import load_cards, get_training_recommendation


# ============================================================
# Scoring Weights
# ============================================================

# Task → 标签偏好映射
TASK_TAG_PREFS = {
    "chat":       ["中文", "对话", "多语言"],
    "code":       ["代码", "编程"],
    "translation":["多语言", "中英双语"],
    "customer":   ["中文", "客服"],
    "safety":     ["中文", "安全"],
    "general":    ["中文", "对话"],
}

# Language → 模型族偏好
LANGUAGE_PREFS = {
    "zh":   ["qwen2", "chatglm", "yi", "intern2", "deepseek"],
    "en":   ["llama", "mistral", "gemma", "phi"],
    "mixed":["qwen2", "deepseek", "yi", "llama"],
}

# User level → max model size
LEVEL_MAX_PARAMS = {
    "beginner": 7.0,   # ≤7B
    "intermediate": 14.0,  # ≤14B
    "advanced": 70.0,  # ≤70B
    "expert": 999.0,   # 不限
}


# ============================================================
# Recommendation
# ============================================================

@dataclass
class ModelRecommendation:
    """推荐结果"""
    display_name: str
    model_id: str
    source_platform: str  # "modelscope" or "huggingface"
    score: float  # 0-100
    reason: str
    card: dict | None
    training_config: dict | None
    rank: int = 0


def recommend_models(
    task: str = "chat",
    language: str = "zh",
    level: str = "intermediate",
    available_vram_gb: float | None = None,
    max_results: int = 5,
) -> list[ModelRecommendation]:
    """
    根据用户条件推荐最佳模型。

    Args:
        task: 任务类型（chat/code/translation/customer/safety/general）
        language: 语言偏好（zh/en/mixed）
        level: 用户级别（beginner/intermediate/advanced/expert）
        available_vram_gb: 可用显存（None 则自动检测）
        max_results: 最多返回几个推荐

    Returns:
        排序后的推荐列表
    """
    # Auto-detect VRAM
    if available_vram_gb is None:
        try:
            info = get_system_info()
            available_vram_gb = info.max_single_vram_gb if info.has_gpu else 8.0
        except Exception:
            available_vram_gb = 8.0

    max_params = LEVEL_MAX_PARAMS.get(level, 14.0)
    cards = load_cards()
    results = []

    # Preferred families for this language
    preferred_families = LANGUAGE_PREFS.get(language, LANGUAGE_PREFS["zh"])
    # Preferred tags for this task
    preferred_tags = TASK_TAG_PREFS.get(task, TASK_TAG_PREFS["chat"])

    for card in cards:
        # Filter: param size
        if card["params_b"] > max_params:
            continue

        # Filter: VRAM
        training = card["training"]
        min_vram = training["qlora"]["min_vram_gb"]
        if min_vram > available_vram_gb:
            continue

        # ---- Scoring ----
        score = 0.0

        # 1. Language match (35 points max)
        family = card["family"]
        if family in preferred_families:
            idx = preferred_families.index(family)
            score += 35 - idx * 7  # First choice = 35, second = 28, etc.
        elif family not in LANGUAGE_PREFS.get("en", []):
            score += 10  # Neutral

        # 2. Task match (30 points max)
        tags = card.get("tags", [])
        tag_match_count = sum(1 for t in preferred_tags if t in tags)
        score += min(30, tag_match_count * 10)

        # 3. VRAM fit (20 points max)
        vram_ratio = min_vram / max(available_vram_gb, 1)
        if vram_ratio <= 0.3:
            score += 20  # Very comfortable
        elif vram_ratio <= 0.5:
            score += 15  # Good fit
        elif vram_ratio <= 0.7:
            score += 10  # Tight
        else:
            score += 5   # Barely fits

        # 4. User level match (10 points max)
        if level == "beginner":
            if card["params_b"] <= 3.0:
                score += 10
            elif card["params_b"] <= 7.0:
                score += 5
        elif level == "intermediate":
            if 3.0 <= card["params_b"] <= 14.0:
                score += 10
            else:
                score += 5
        elif level == "advanced":
            if 7.0 <= card["params_b"] <= 40.0:
                score += 10
            else:
                score += 5

        # 5. Source availability (5 points)
        sources = card.get("sources", [])
        if any(s["platform"] == "modelscope" for s in sources):
            score += 3  # 国内访问更快
        if any(s["platform"] == "huggingface" for s in sources):
            score += 2

        # Get training recommendation
        best_source = sources[0] if sources else {"platform": "huggingface", "id": card["display_name"]}
        training_rec = get_training_recommendation(
            best_source["id"], available_vram_gb
        )

        results.append(ModelRecommendation(
            display_name=card["display_name"],
            model_id=best_source["id"],
            source_platform=best_source["platform"],
            score=round(score, 1),
            reason=_build_reason(card, score, available_vram_gb, language, task),
            card=card,
            training_config=training_rec,
        ))

    # Sort by score
    results.sort(key=lambda x: x.score, reverse=True)

    # Assign ranks
    for i, r in enumerate(results[:max_results]):
        r.rank = i + 1

    return results[:max_results]


def _build_reason(card: dict, score: float, vram_gb: float, language: str, task: str) -> str:
    """生成推荐理由"""
    parts = []

    if score >= 85:
        parts.append("🏆 最佳匹配")
    elif score >= 70:
        parts.append("👍 强烈推荐")
    elif score >= 50:
        parts.append("✓ 可选方案")

    # Language
    lang_map = {"zh": "中文", "en": "英文", "mixed": "中英双语"}
    if language in ["zh", "mixed"] and any(t in card.get("tags", []) for t in ["中文", "中英双语"]):
        parts.append(f"中文能力强")
    elif language == "en" and "英文" in card.get("tags", []):
        parts.append("英文能力强")

    # Size
    parts.append(f"{card['params_b']}B 参数")

    # VRAM
    min_vram = card["training"]["qlora"]["min_vram_gb"]
    if vram_gb >= min_vram * 2:
        parts.append(f"显存充裕（需 {min_vram}GB，有 {vram_gb:.0f}GB）")
    else:
        parts.append(f"显存刚好（需 {min_vram}GB）")

    return " · ".join(parts)


def get_quick_recommendation() -> dict:
    """
    快速推荐：自动检测硬件后返回最佳模型。

    Returns:
        {
            "top_model": ModelRecommendation (as dict),
            "all_recommendations": list,
            "hardware_summary": str,
        }
    """
    info = get_system_info()
    vram = info.max_single_vram_gb if info.has_gpu else 8.0

    # Default: Chinese chat, intermediate
    recs = recommend_models(
        task="chat",
        language="zh",
        level="intermediate" if vram >= 8 else "beginner",
        available_vram_gb=vram,
        max_results=5,
    )

    hardware_summary = f"GPU: {info.gpus[0].name if info.has_gpu else 'N/A'} · 显存: {vram:.1f}GB · RAM: {info.ram_total_gb:.1f}GB"

    return {
        "top_model": {
            "name": recs[0].display_name if recs else "N/A",
            "model_id": recs[0].model_id if recs else "",
            "score": recs[0].score if recs else 0,
            "reason": recs[0].reason if recs else "",
        } if recs else None,
        "all_recommendations": [
            {"rank": r.rank, "name": r.display_name, "score": r.score, "reason": r.reason}
            for r in recs
        ],
        "hardware_summary": hardware_summary,
    }


def recommend_for_task(task: str, language: str = "zh") -> list[dict]:
    """为特定任务推荐模型（简化接口，前端用）"""
    recs = recommend_models(task=task, language=language)
    return [
        {
            "model": r.display_name,
            "model_id": r.model_id,
            "params": f"{r.card['params_b']}B" if r.card else "?",
            "vram": r.card["training"]["qlora"]["min_vram_gb"] if r.card else "?",
            "score": r.score,
            "reason": r.reason,
            "source": r.source_platform,
        }
        for r in recs
    ]


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 智能模型推荐引擎测试")
    print("=" * 60)

    # Quick recommendation
    quick = get_quick_recommendation()
    print(f"\n📊 硬件: {quick['hardware_summary']}")
    print(f"\n🏆 最佳推荐: {quick['top_model']['name'] if quick['top_model'] else 'N/A'}")
    print(f"   理由: {quick['top_model']['reason'] if quick['top_model'] else ''}")

    # Per-task recommendations
    for task, label in [("chat", "💬 对话"), ("code", "💻 代码"), ("translation", "🌐 翻译")]:
        print(f"\n{label}任务推荐:")
        recs = recommend_for_task(task, "zh")
        for r in recs[:3]:
            bar = "█" * (r["score"] // 10)
            print(f"  {r['rank']}. {r['model']} ({r['params']}) [score:{r['score']}] {bar}")
            print(f"     {r['reason']}")

    print()
    print("✅ 模型推荐引擎自检完成")
