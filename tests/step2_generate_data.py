"""
实操验证 Step 2: 生成训练数据集

生成 500 条中文对话训练数据，覆盖多个主题
使用 ShareGPT 格式（标准的多轮对话格式）
"""

import json
import random
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "datasets" / "demo-zh-conversation"
OUTPUT_FILE = OUTPUT_DIR / "data.json"

# ============================================================
# 中文对话模板
# ============================================================

TEMPLATES = [
    # 日常对话
    {"user": "你好！今天天气真好。", "assistant": "是啊！阳光明媚，特别适合出去走走。你今天有什么计划吗？"},
    {"user": "最近心情不太好，工作压力有点大。", "assistant": "我能理解你的感受。工作压力确实会让人很疲惫。建议你可以试试：1）每天抽10分钟深呼吸放松；2）下班后做些喜欢的事转移注意力；3）和信任的朋友聊聊天。照顾好自己很重要！"},
    {"user": "能给我推荐几道简单好做的家常菜吗？", "assistant": "当然可以！推荐几道新手友好又好吃的：\n1. 番茄炒蛋 - 经典中的经典，酸甜开胃\n2. 青椒肉丝 - 火候掌握好，肉嫩菜脆\n3. 麻婆豆腐 - 用现成的麻婆酱，5分钟搞定\n4. 可乐鸡翅 - 香甜可口，几乎零失败\n5. 蒜蓉西兰花 - 清爽健康，焯水拌蒜就行\n\n需要哪道菜的详细做法吗？"},
    {"user": "我想学一门新技能，有什么建议吗？", "assistant": "这取决于你的兴趣和目标！以下是几个热门方向：\n\n📊 数据分析：SQL + Python，市场需求大\n🎨 设计：Figma + 设计思维，创造性强\n💻 编程：Python 或 JavaScript，应用最广\n📝 写作：自媒体、文案，表达能力的提升\n\n建议先想清楚：你是想转行、副业还是纯粹兴趣？我可以针对性地给你规划学习路线。"},

    # 知识问答
    {"user": "什么是量子计算？用通俗的语言解释一下。", "assistant": "量子计算是一种全新的计算范式。用最简单的比喻来说：\n\n传统计算机用「比特（bit）」来表示信息，每个比特只能是 0 或 1，就像开关一样，只能处于一个状态。\n\n量子计算机则使用「量子比特（qubit）」，它有一个神奇的特性叫「叠加态」——一个量子比特可以同时是 0 和 1。\n\n这听起来很玄乎，但可以这样理解：传统计算机在一个迷宫里一次只能尝试一条路，量子计算机则可以同时尝试所有路径。\n\n所以在解决某些特定问题（如大数分解、药物分子模拟、优化问题）时，量子计算有巨大的速度优势。"},
    {"user": "介绍一下中国四大名著的作者和主要内容。", "assistant": "中国四大名著是中华文学的瑰宝，简单介绍如下：\n\n📕 《三国演义》- 罗贯中\n讲述东汉末年到西晋初年魏、蜀、吴三国的政治军事斗争。群雄逐鹿、英雄辈出，核心是「忠义」二字。经典人物：诸葛亮、关羽、曹操。\n\n📗 《水浒传》- 施耐庵\n北宋末年 108 位好汉被逼上梁山、聚义抗争的故事。展现了「官逼民反」的主题。经典人物：宋江、武松、林冲。\n\n📘 《西游记》- 吴承恩\n唐僧师徒四人历经九九八十一难西天取经的神话故事。充满想象力，隐喻人生修行。经典人物：孙悟空、唐僧、猪八戒。\n\n📙 《红楼梦》- 曹雪芹\n以贾宝玉、林黛玉的爱情悲剧为主线，展现封建大家族兴衰的社会百态。被誉为「中国封建社会的百科全书」。经典人物：贾宝玉、林黛玉、王熙凤。"},
    {"user": "能解释一下 ChatGPT 是怎么工作的吗？", "assistant": "ChatGPT 的工作原理可以分几个层面来理解：\n\n**1. 底层架构：Transformer + 自回归**\n它基于 Transformer 架构，通过预测「下一个词」来生成文本。给定上文，模型计算下一个最可能出现的词是什么。\n\n**2. 训练三步曲**\n- 预训练（Pre-training）：在海量文本上学习语言规律，学语法、事实、推理模式\n- 监督微调（SFT）：用人工标注的高质量问答数据教模型如何「对话」\n- RLHF（人类反馈强化学习）：人类对模型的不同回答排序，教会模型什么是「好回答」\n\n**3. 为什么它这么聪明？**\n本质上它是「概率统计 + 海量数据 + 超强算力」的产物。它没有真正的理解，但学到的模式足以应对绝大多数问题。"},

    # 代码与编程
    {"user": "Python 中 list 和 tuple 有什么区别？", "assistant": "Python 中 list 和 tuple 的核心区别：\n\n| 特性 | list | tuple |\n|------|------|-------|\n| 可变性 | 可变（可增删改） | 不可变（创建后不能改） |\n| 语法 | `[1, 2, 3]` | `(1, 2, 3)` |\n| 性能 | 稍慢，内存占用更大 | 稍快，内存占用更小 |\n| 用途 | 存储可变数据集合 | 存储不变数据、作为字典键 |\n\n最佳实践：\n- 需要增删改元素 → 用 list\n- 数据不需要改变（如坐标、配置） → 用 tuple\n- 函数返回多个值 → 用 tuple\n- 作为 dict 的 key → 用 tuple（list 不可哈希）"},
    {"user": "帮我写一个快速排序的 Python 实现。", "assistant": "这是快速排序的 Python 实现，附带注释：\n\n```python\ndef quick_sort(arr):\n    \"\"\"快速排序（原地排序版本）\"\"\"\n    def _partition(low, high):\n        pivot = arr[high]  # 选最后一个元素为基准\n        i = low - 1\n        for j in range(low, high):\n            if arr[j] <= pivot:\n                i += 1\n                arr[i], arr[j] = arr[j], arr[i]\n        arr[i + 1], arr[high] = arr[high], arr[i + 1]\n        return i + 1\n    \n    def _quick_sort(low, high):\n        if low < high:\n            pi = _partition(low, high)\n            _quick_sort(low, pi - 1)\n            _quick_sort(pi + 1, high)\n    \n    _quick_sort(0, len(arr) - 1)\n    return arr\n\n# 测试\nprint(quick_sort([3, 6, 8, 10, 1, 2, 1]))  # [1, 1, 2, 3, 6, 8, 10]\n```\n\n时间复杂度：平均 O(n log n)，最坏 O(n²)。空间复杂度 O(log n)。"},

    # 写作助手
    {"user": "帮我写一段产品推荐文案，推销一款智能手表。", "assistant": "当然！以下是产品文案：\n\n---\n\n**⌚ 掌控你的每一天 · ProFit 智能手表**\n\n不是所有手表都懂你的节奏。\n\n✨ **全天候健康守护**\n血氧监测 · 心率预警 · 睡眠分析 · 压力指数\n24小时默默关注，比你自己更了解你的身体。\n\n⚡ **7天超长续航**\n告别每日充电焦虑，一周只充一次。\n\n🏃 **50+ 运动模式**\n跑步、游泳、骑行、瑜伽...\n每一次流汗，都被精准记录。\n\n📱 **智慧生活，一触即达**\n来电提醒 · 消息速回 · NFC支付 · 语音助手\n抬手之间，世界尽在掌握。\n\n🔥 首发特惠：¥999（原价 ¥1,299）\n立即下单，赠送定制表带 ×2！\n\n---\n\n需要调整风格或突出不同卖点吗？"},
    {"user": "给新人写一段婚礼致辞。", "assistant": "以下是一段温馨真挚的婚礼致辞：\n\n---\n\n各位亲朋好友，大家好！\n\n今天阳光正好，我们在聚在这里，见证 [新郎] 和 [新娘] 人生中最美好的时刻。\n\n有人说，人这一生会遇到约 2920 万人，而两个人相爱的概率只有 0.000049。所以，你们能站在这里，本身就是奇迹。\n\n[新郎]，我认识你这么多年，见证了你从一个毛头小子成长为今天这个有担当的男人。\n[新娘]，你的温暖和善良让身边所有人都如沐春风。\n\n今天不是终点，是起点。婚姻不是童话里的「从此幸福生活」，而是两个人并肩面对生活的一切——柴米油盐也好，风雨挑战也罢，都有人牵着你的手说「有我」\n\n在此，我提议大家举杯：\n祝愿这对新人\n执子之手，与子偕老\n朝朝暮暮，岁岁年年！\n\n干杯！🍷\n\n---\n\n（可以根据实际情况调整人名和细节）"},
]

# ============================================================
# 生成变体：对每个模板做同义改写和细微调整
# ============================================================

def generate_variations(template, count=3):
    """对一条模板生成多个变体"""
    variations = []
    user_text = template["user"]
    assistant_text = template["assistant"]

    # 变体策略：稍微修改提问方式
    user_prefixes = [
        "请问", "能告诉我", "帮我", "请教一下", "想问一下",
        "你好，我想知道", "可以介绍一下吗？", "", "你好，"
    ]

    for i in range(count):
        if i == 0:
            # Original
            var_user = user_text
            var_assistant = assistant_text
        else:
            # Slightly varied
            prefix = random.choice(user_prefixes)
            var_user = prefix + user_text if prefix else user_text
            var_assistant = assistant_text  # 回答不变

        variations.append({
            "messages": [
                {"role": "user", "content": var_user},
                {"role": "assistant", "content": var_assistant}
            ]
        })

    return variations


def main():
    """主函数：生成 500 条训练数据"""
    print("=" * 60)
    print("实操验证 Step 2: 生成训练数据集")
    print("=" * 60)

    data = []

    # 1. 从模板生成变体
    print(f"  模板数量: {len(TEMPLATES)}")
    for template in TEMPLATES:
        variations = generate_variations(template, count=12)
        data.extend(variations)

    print(f"  模板生成: {len(data)} 条")

    # 2. 如果不足 500 条，用模板混搭补充
    while len(data) < 500:
        # 随机取两条不同的模板，混搭 user 和 assistant
        t1 = random.choice(TEMPLATES)
        t2 = random.choice(TEMPLATES)
        if t1 == t2:
            continue
        data.append({
            "messages": [
                {"role": "user", "content": t1["user"]},
                {"role": "assistant", "content": t2["assistant"]}
            ]
        })

    # 截断到 500
    data = data[:500]
    random.shuffle(data)

    # 3. 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 4. 保存元信息
    meta = {
        "name": "demo-zh-conversation",
        "format": "sharegpt",
        "record_count": len(data),
        "topics": ["日常对话", "知识问答", "编程", "写作"],
        "language": "zh",
        "description": "自动生成的 500 条中文对话数据，用于实操验证微调流程",
        "created_by": "模型训练室数据生成器",
    }
    with open(OUTPUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 5. 统计
    total_chars = sum(
        sum(len(msg["content"]) for msg in item["messages"])
        for item in data
    )
    avg_user_len = sum(
        len(item["messages"][0]["content"])
        for item in data
    ) / len(data)
    avg_asst_len = sum(
        len(item["messages"][1]["content"])
        for item in data
    ) / len(data)

    print(f"  最终条数: {len(data)}")
    print(f"  总字符数: {total_chars:,}")
    print(f"  平均用户输入长度: {avg_user_len:.0f} 字符")
    print(f"  平均助手输出长度: {avg_asst_len:.0f} 字符")
    print(f"  文件大小: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print(f"  保存路径: {OUTPUT_FILE}")
    print()
    print("  ✅ 数据集生成完毕")

    # 6. 预览 3 条
    print()
    print("  📋 预览（前 3 条）:")
    for i, item in enumerate(data[:3]):
        user = item["messages"][0]["content"][:80]
        assistant = item["messages"][1]["content"][:80]
        print(f"  [{i+1}] Q: {user}...")
        print(f"      A: {assistant}...")


if __name__ == "__main__":
    main()
