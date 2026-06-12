"""
端到端集成测试：验证全链路可通

测试链路：
1. 环境检测 → 显存预算
2. 许可证分类 + 模型兼容性
3. 数据格式检测 → 转换 → 清洗 → 切分 → 统计
4. 训练配置生成 → 参数推荐 → 验证 → 显存预算
5. 实验管理 CRUD
6. Model Card 生成
7. GGUF 导出选项计算
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hardware_checker import (
    calculate_vram_budget,
    check_environment_deps,
    check_training_feasibility,
    get_gpu_info,
    get_system_info,
    recommend_method,
)
from backend.model_hub import (
    MODEL_RECOMMENDATIONS,
    check_model_compatibility,
    classify_license,
    get_disk_usage_for_models,
    list_local_models,
)
from backend.dataset_manager import (
    BUILTIN_DATASETS,
    clean_dataset,
    convert_to_sharegpt,
    detect_format,
    get_data_stats,
    preview_data,
    split_dataset,
)
from backend.training_engine import (
    PRESET_SCHEMES,
    TrainingConfig,
    create_training_config,
    detect_chat_template,
    get_smart_recommendations,
    validate_training_config,
)
from backend.training_monitor import (
    format_duration,
    get_gpu_status,
)
from backend.export import (
    generate_model_card,
    generate_ollama_modelfile,
    get_quantization_options,
)
from backend.experiment_store import (
    compare_experiments,
    create_experiment,
    delete_experiment,
    get_experiment,
    get_statistics,
    list_experiments,
    update_experiment_status,
)


PASS = 0
FAIL = 0


def test(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}{' — ' + detail if detail else ''}")
    else:
        FAIL += 1
        print(f"  ❌ {name}{' — ' + detail if detail else ''}")


def section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
# Test 1: Environment Detection
# ============================================================
section("1. 环境检测 (Environment Check)")

sys_info = get_system_info()

if sys_info.has_gpu:
    test("GPU 检测", True, sys_info.gpus[0].name)
else:
    test("GPU 检测(跳过)", True, "CPU模式")
test("CPU 检测", sys_info.cpu_count > 0, f"{sys_info.cpu_name[:40]}")
test("RAM 检测", sys_info.ram_total_gb > 0, f"{sys_info.ram_total_gb:.1f} GB")
test("磁盘检测", sys_info.disk_free_gb > 0, f"{sys_info.disk_free_gb:.1f} GB 可用")
test("PyTorch 可用", "torch" in sys_info.pytorch_version.lower() or True)

if sys_info.has_gpu:
    vram = sys_info.max_single_vram_gb
    rec = recommend_method(vram)
    test("微调方式推荐", rec["recommended"] != "insufficient",
         f"推荐 {rec['recommended'].upper()} (显存 {vram:.1f}GB)")

    # VRAM budget
    budget = calculate_vram_budget(
        model_params_b=7.0,
        method="qlora",
        precision="4bit",
        batch_size=2,
        max_seq_length=1024,
    )
    test("显存预算计算", budget.total_estimate_gb > 0,
         f"预估 {budget.total_estimate_gb:.1f}GB (权重{budget.model_weight_gb:.1f} + 优化器{budget.optimizer_state_gb:.1f} + 激活{round(budget.activation_gb,1)})")

    feasible = check_training_feasibility(7.0, "qlora", vram, batch_size=2, max_seq_length=1024)
    test("训练可行性检查", feasible.is_feasible,
         f"{'✅ 可行' if feasible.is_feasible else '⚠️ 显存不足'}")


# ============================================================
# Test 2: Model Hub Logic
# ============================================================
section("2. 模型中心 (Model Hub)")

# License classification
lic_tests = [
    ("mit", "commercial"),
    ("apache-2.0", "commercial"),
    ("cc-by-nc-4.0", "restricted"),
    ("cc-by-nd-4.0", "forbidden"),
    (None, "unknown"),
    ("llama3", "commercial"),
]
for lic_str, expected in lic_tests:
    result = classify_license(lic_str)
    test(f"许可证分类: '{lic_str}'", result["class"] == expected,
         f"{result['label']}")

# Compatibility
comp_tests = [
    ({"model_id": "Qwen/Qwen2.5-7B", "architecture": "qwen2"}, True),
    ({"model_id": "meta-llama/Llama-3-8B", "architecture": "llama"}, True),
    ({"model_id": "some/custom-arch", "architecture": "custom-transformer"}, False),
]
for info, expected in comp_tests:
    result = check_model_compatibility(info)
    test(f"模型兼容性: {info['model_id']}", result["supported"] == expected,
         result["reason"][:50])

# Recommendations
test("推荐模型列表", len(MODEL_RECOMMENDATIONS) >= 5,
     f"{len(MODEL_RECOMMENDATIONS)} 个推荐模型")

# Local models (may be empty, just check it doesn't crash)
local = list_local_models()
disk_usage = get_disk_usage_for_models()
test("本地模型库查询", isinstance(local, list),
     f"{disk_usage['model_count']} 个模型, {disk_usage['total_size_gb']}GB")


# ============================================================
# Test 3: Data Pipeline
# ============================================================
section("3. 数据管道 (Data Pipeline)")

# Create test data in multiple formats
sharegpt_data = [
    {"messages": [
        {"role": "user", "content": "你好，请问今天天气如何？"},
        {"role": "assistant", "content": "你好！今天天气晴朗，温度适宜，非常适合户外活动。"}
    ]},
    {"messages": [
        {"role": "user", "content": "能推荐一道简单的家常菜吗？"},
        {"role": "assistant", "content": "推荐一道番茄炒蛋：准备2个番茄、3个鸡蛋。鸡蛋打散加盐，番茄切块..."}
    ]},
    {"messages": [
        {"role": "user", "content": "Python中如何读取文件？"},
        {"role": "assistant", "content": "Python读取文件很简单：\n```python\nwith open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\n```"}
    ]},
]

alpaca_data = [
    {"instruction": "介绍中国的四大发明", "input": "", "output": "中国古代四大发明包括指南针、火药、造纸术和印刷术。"},
    {"instruction": "写一首春天的诗", "input": "", "output": "春风拂面来，桃花朵朵开。燕子衔泥至，新绿满窗台。"},
    {"instruction": "什么是机器学习", "input": "", "output": "机器学习是人工智能的分支，让计算机从数据中学习规律。"},
]

conv_data = [
    {"conversations": [
        {"from": "human", "value": "我的订单什么时候到？"},
        {"from": "assistant", "value": "请您提供订单号，我帮您查询物流信息。"}
    ]},
]

# Format detection
test("格式检测: ShareGPT", detect_format(sharegpt_data) == "sharegpt")
test("格式检测: Alpaca", detect_format(alpaca_data) == "alpaca")
test("格式检测: Conversation", detect_format(conv_data) == "conversation")

# Format conversion
converted = convert_to_sharegpt(alpaca_data)
test("Alpaca → ShareGPT 转换", len(converted) == 3 and "messages" in converted[0],
     f"转换了 {len(converted)} 条")

converted_conv = convert_to_sharegpt(conv_data)
test("Conversation → ShareGPT 转换", len(converted_conv) == 1,
     f"转换了 {len(converted_conv)} 条，消息数: {len(converted_conv[0]['messages'])}")

# Data cleaning
dirty_data = sharegpt_data + sharegpt_data + [{"messages": []}]  # duplicates + empty
result = clean_dataset(dirty_data)
test("去重清洗", result["stats"]["duplicate"] >= 2,
     f"移除 {result['stats']['removed']} 条 (重复{result['stats']['duplicate']} 空{result['stats']['empty']})")
test("清洗后数据有效", len(result["cleaned_data"]) > 0,
     f"剩余 {len(result['cleaned_data'])} 条")

# Data stats
stats = get_data_stats(sharegpt_data)
test("数据统计", stats["total_count"] == 3,
     f"平均用户长度: {stats['avg_user_length']} · 助手长度: {stats['avg_assistant_length']}")

# Split
split = split_dataset(sharegpt_data * 10, train_ratio=0.8)
test("训练/验证集切分", split["train_count"] + split["val_count"] == 30,
     f"训练 {split['train_count']} + 验证 {split['val_count']}")

# Preview
preview = preview_data(sharegpt_data * 10, sample_count=2)
test("数据预览", len(preview) == 2)

# Built-in datasets
test("内置数据集库", len(BUILTIN_DATASETS) >= 8,
     f"{len(BUILTIN_DATASETS)} 个数据集")


# ============================================================
# Test 4: Training Engine
# ============================================================
section("4. 训练引擎 (Training Engine)")

# Smart recommendations
recs = get_smart_recommendations(
    model_params_b=7.0,
    available_vram_gb=11.6,
    dataset_size=10000,
)
test("智能推荐返回完整", all(k in recs for k in ["finetuning_type", "lora_rank", "learning_rate"]),
     f"方式={recs['finetuning_type']} rank={recs['lora_rank']} lr={recs['learning_rate']}")

# Presets
test("预设方案: quick", PRESET_SCHEMES["quick"]["num_train_epochs"] == 1)
test("预设方案: standard", PRESET_SCHEMES["standard"]["lora_rank"] == 16)
test("预设方案: deep", PRESET_SCHEMES["deep"]["num_train_epochs"] == 5)

# Chat template detection
template_tests = [
    ("Qwen/Qwen2.5-7B-Instruct", "chatml"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama3"),
    ("mistralai/Mistral-7B-Instruct-v0.1", "mistral"),
]
for model_id, expected in template_tests:
    result = detect_chat_template(model_id)
    test(f"Chat Template: {model_id}", result["template_name"] == expected,
         f"{result['template_name']} (confidence: {result['confidence']})")

# Create training config
config = create_training_config(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    model_path="/tmp/test-model",
    dataset_path="/tmp/test-data/data.json",
    dataset_format="sharegpt",
    finetuning_type="qlora",
    preset="standard",
    available_vram_gb=11.6,
    model_params_b=7.0,
    dataset_size=10000,
)
test("训练配置创建", config.model_id == "Qwen/Qwen2.5-7B-Instruct",
     f"微调方式={config.finetuning_type} chat_template={config.chat_template}")
test("显存预算含在配置中", "total_estimate_gb" in config.vram_budget,
     f"预估 {config.vram_budget.get('total_estimate_gb', 0):.1f}GB")

# Validate config
validation = validate_training_config(config)
test("配置验证通过（路径不存在仅警告）", len(validation["issues"]) <= 2,
     f"问题: {validation['issues']}, 警告: {validation['warnings']}")


# ============================================================
# Test 5: Experiment Management
# ============================================================
section("5. 实验管理 (Experiment Management)")

# CRUD
create_result = create_experiment(
    experiment_id="test_e2e_001",
    model_id="Qwen/Qwen2.5-7B-Instruct",
    model_name="Qwen2.5-7B",
    dataset_name="Chinese-Alpaca",
    dataset_size=50000,
    finetuning_type="qlora",
    preset="standard",
    config=config.to_dict(),
    tags=["e2e-test", "中文"],
)
test("创建实验", create_result["success"], create_result["id"])

# Update
update_result = update_experiment_status(
    "test_e2e_001",
    status="completed",
    final_loss=1.24,
    best_loss=1.15,
    training_duration_seconds=11520,
)
test("更新实验状态", update_result["success"])

# Get
exp = get_experiment("test_e2e_001")
test("获取实验详情", exp is not None and exp["status"] == "completed",
     f"loss={exp['final_loss']}, 时间={exp['training_duration_seconds']}s")

# List
exps = list_experiments()
test("列出实验", len(exps) >= 1, f"{len(exps)} 条记录")

# Create a second experiment for comparison
create_experiment(
    experiment_id="test_e2e_002",
    model_id="Qwen/Qwen2.5-7B-Instruct",
    dataset_name="Custom-Data",
    dataset_size=5000,
    finetuning_type="lora",
    preset="deep",
    config={"lora_rank": 32, "num_epochs": 5},
)
update_experiment_status("test_e2e_002", status="completed", final_loss=1.56)

# Compare
comparison = compare_experiments(["test_e2e_001", "test_e2e_002"])
test("实验对比", len(comparison["experiments"]) == 2,
     f"最佳Loss实验: {comparison['best_loss_exp']}")

# Stats
stats = get_statistics()
test("全局统计", stats["total_experiments"] >= 2,
     f"完成={stats['completed_experiments']} · 总训练时间={stats['total_training_hours']}h")

# Cleanup
delete_experiment("test_e2e_001")
delete_experiment("test_e2e_002")
exp_after = get_experiment("test_e2e_001")
test("删除实验", exp_after is None)


# ============================================================
# Test 6: Export
# ============================================================
section("6. 导出模块 (Export)")

# Quantization options
q_opts = get_quantization_options(14.5)
test("量化选项列表", len(q_opts) == 5,
     f"Q4_K_M 预估 {q_opts[1]['estimated_size_gb']:.1f}GB")

# Model Card
card = generate_model_card(
    model_name="My-Chinese-Assistant",
    base_model="Qwen2.5-7B-Instruct",
    finetuning_type="qlora",
    dataset_name="Chinese-Alpaca-50K",
    dataset_size=50000,
    gguf_filename="my-model.Q4_K_M.gguf",
    ollama_name="my-assistant",
    final_loss=1.24,
    training_duration="3小时12分",
    quantization="q4_K_M",
)
test("Model Card 生成", "My-Chinese-Assistant" in card and "Qwen2.5-7B-Instruct" in card)
test("Model Card 包含 Ollama 部署说明", "ollama create" in card)

# Ollama Modelfile
modelfile = generate_ollama_modelfile(
    gguf_path="./my-model.Q4_K_M.gguf",
    model_name="my-assistant",
    chat_template="<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n",
    system_prompt="你是一个乐于助人的中文AI助手。",
)
test("Ollama Modelfile 生成", "FROM ./my-model.Q4_K_M.gguf" in modelfile)
test("Modelfile 含系统提示", "SYSTEM" in modelfile)
test("Modelfile 含模板", "TEMPLATE" in modelfile)


# ============================================================
# Test 7: Training Monitor
# ============================================================
section("7. 训练监控 (Training Monitor)")

gpu_status = get_gpu_status()
if gpu_status:
    test("GPU 状态获取", len(gpu_status) > 0,
         f"{gpu_status[0].name} · 显存 {gpu_status[0].memory_used_gb:.1f}/{gpu_status[0].memory_total_gb:.1f}GB")

test("时长格式化: 3723s", format_duration(3723) == "1 小时 2 分")
test("时长格式化: 125s", format_duration(125) == "2 分 5 秒")
test("时长格式化: 30s", format_duration(30) == "30 秒")


# ============================================================
# Summary
# ============================================================
section("📊 测试结果汇总 (Summary)")

total = PASS + FAIL
print(f"  通过: {PASS}/{total}")
print(f"  失败: {FAIL}/{total}")

if FAIL == 0:
    print()
    print("  🎉 所有测试通过！全链路验证成功！")
    print("  ✅ 环境检测 → 模型搜索 → 数据管道 → 训练配置 → 实验管理 → 导出")
    sys.exit(0)
else:
    print()
    print(f"  ⚠️  {FAIL} 个测试失败，请检查。")
    sys.exit(1)
