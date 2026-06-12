"""
全覆盖功能测试 —— 模型训练室 v0.2

测试范围：
1. 环境检测 (hardware_checker)
2. 模型中心 (model_hub) — 搜索/许可证/兼容性/本地库
3. 数据管道 (dataset_manager) — 检测/转换/清洗/切分/统计/内置库
4. 训练引擎 (training_engine) — 参数推荐/预设/Chat Template/配置生成/验证
5. 训练监控 (training_monitor) — GPU状态/日志解析/格式化
6. 导出模块 (export) — 量化选项/Model Card/Ollama Modelfile
7. 实验管理 (experiment_store) — CRUD/对比/统计
8. 模型卡片 (model_cards) — 匹配/推荐/搜索
9. 前端构建 (app) — Gradio Blocks 加载
10. GGUF量化 (quantize_gguf) — 脚本可用性
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = 0
FAIL = 0
WARN = 0


def ok(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}{' — ' + detail if detail else ''}")
    else:
        FAIL += 1
        print(f"  ❌ {name}{' — ' + detail if detail else ''}")


def warn(name, detail=""):
    global WARN
    WARN += 1
    print(f"  ⚠️  {name}{' — ' + detail if detail else ''}")


def section(title):
    print()
    print("━" * 60)
    print(f"  {title}")
    print("━" * 60)


# ============================================================
# Test 1: 环境检测
# ============================================================
section("1️⃣  环境检测 (hardware_checker)")

from backend.hardware_checker import (
    SystemInfo, VRAMBudget, GPUInfo,
    get_gpu_info, get_system_info, check_environment_deps,
    calculate_vram_budget, recommend_method,
    check_training_feasibility, generate_env_report,
)

info = get_system_info()
ok("GPU检测", len(info.gpus) > 0, f"{info.gpus[0].name if info.gpus else 'N/A'}")
ok("CPU检测", info.cpu_count > 0)
ok("RAM检测", info.ram_total_gb > 0, f"{info.ram_total_gb:.1f}GB")
ok("磁盘检测", info.disk_free_gb > 0, f"{info.disk_free_gb:.1f}GB可用")
ok("PyTorch版本", len(info.pytorch_version) > 0, info.pytorch_version[:20])
ok("CUDA可用检查", isinstance(info.cuda_available, bool))

if info.has_gpu:
    vram = info.max_single_vram_gb
    rec = recommend_method(vram)
    ok("微调方式推荐", rec["recommended"] != "insufficient", f"推荐{rec['recommended'].upper()} ({vram:.1f}GB)")
    ok("推荐包含选项", len(rec["options"]) > 0)

    # VRAM budget
    for method in ["qlora", "lora", "full"]:
        budget = calculate_vram_budget(7.0, method=method, batch_size=2, max_seq_length=2048)
        ok(f"显存预算({method})", budget.total_estimate_gb > 0,
           f"{budget.total_estimate_gb:.1f}GB (权重{budget.model_weight_gb:.1f}+优化器{budget.optimizer_state_gb:.1f})")

    feasible = check_training_feasibility(7.0, "qlora", vram, batch_size=2, max_seq_length=1024)
    ok("训练可行性检查", feasible.is_feasible)
    ok("可行性含警告字段", "warning" in feasible.__dict__ or hasattr(feasible, 'warning'))

deps = check_environment_deps()
ok("依赖检查", len(deps) >= 5, f"{len(deps)}项")
ok("PyTorch已安装", deps.get("torch", {}).get("installed", False))
ok("CUDA已安装", deps.get("cuda", {}).get("installed", False))

report = generate_env_report()
ok("环境报告生成", len(report) > 500, f"{len(report)}字符")
ok("报告含关键信息", "GPU" in report and "显存" in report)


# ============================================================
# Test 2: 模型中心
# ============================================================
section("2️⃣  模型中心 (model_hub)")

from backend.model_hub import (
    classify_license, check_model_compatibility,
    MODEL_RECOMMENDATIONS, list_local_models, get_disk_usage_for_models,
    _match_pattern, get_model_detail,
)

# 许可证分类
for lic_str, expected in [
    ("mit", "commercial"), ("apache-2.0", "commercial"),
    ("cc-by-nc-4.0", "restricted"), ("cc-by-nd-4.0", "forbidden"),
    (None, "unknown"), ("llama3", "commercial"), ("qwen", "commercial"),
]:
    r = classify_license(lic_str)
    ok(f"许可证: '{lic_str}'", r["class"] == expected, f"→ {r['label']}")

# 模型兼容性
for info_dict, expected_support in [
    ({"model_id": "Qwen/Qwen2.5-7B", "architecture": "qwen2"}, True),
    ({"model_id": "meta-llama/Llama-3-8B", "architecture": "llama"}, True),
    ({"model_id": "deepseek-ai/DeepSeek-Coder-6.7B", "architecture": "llama"}, True),
    ({"model_id": "mistralai/Mistral-7B", "architecture": "mistral"}, True),
    ({"model_id": "google/gemma-2-2b", "architecture": "gemma2"}, True),
    ({"model_id": "unknown/model", "architecture": "custom-transformer"}, False),
]:
    r = check_model_compatibility(info_dict)
    ok(f"兼容性: {info_dict['model_id']}", r["supported"] == expected_support)

ok("推荐模型列表", len(MODEL_RECOMMENDATIONS) >= 5, f"{len(MODEL_RECOMMENDATIONS)}个")
ok("推荐模型含所有字段", all(all(k in m for k in ["name", "model_id", "params_b", "license"]) for m in MODEL_RECOMMENDATIONS))

# 本地库
local = list_local_models()
ok("本地模型库查询", isinstance(local, list))
disk = get_disk_usage_for_models()
ok("模型磁盘用量", "model_count" in disk and "total_size_gb" in disk)
ok("下载模型已存在", any("Qwen" in m.get("model_id", "") for m in local) or disk["model_count"] > 0,
   f"{disk['model_count']}个模型, {disk['total_size_gb']}GB")

# Pattern matching
ok("Glob匹配: *.safetensors", _match_pattern("model.safetensors", "*.safetensors"))
ok("Glob不匹配: .bin", not _match_pattern("model.bin", "*.safetensors"))


# ============================================================
# Test 3: 数据管道
# ============================================================
section("3️⃣  数据管道 (dataset_manager)")

from backend.dataset_manager import (
    BUILTIN_DATASETS, DatasetInfo,
    detect_format, convert_to_sharegpt, convert_to_alpaca,
    clean_dataset, get_data_stats, preview_data, split_dataset,
    import_file, save_dataset, load_dataset, list_local_datasets,
)

# 格式检测
sharegpt = [{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}]
alpaca = [{"instruction": "写诗", "input": "", "output": "春眠不觉晓"}]
conv = [{"conversations": [{"from": "human", "value": "Hi"}, {"from": "assistant", "value": "Hello"}]}]

ok("检测: ShareGPT", detect_format(sharegpt) == "sharegpt")
ok("检测: Alpaca", detect_format(alpaca) == "alpaca")
ok("检测: Conversation", detect_format(conv) == "conversation")
ok("检测: 空数据", detect_format([]) == "unknown")

# 格式转换
s2a = convert_to_alpaca(sharegpt)
ok("ShareGPT→Alpaca", len(s2a) == 1 and "instruction" in s2a[0])
a2s = convert_to_sharegpt(alpaca)
ok("Alpaca→ShareGPT", len(a2s) == 1 and "messages" in a2s[0])
c2s = convert_to_sharegpt(conv)
ok("Conv→ShareGPT", len(c2s) == 1)

# 数据清洗（使用足够长的文本确保不被 min_length 过滤）
long_text = [{"messages": [
    {"role": "user", "content": "你好，请问今天天气怎么样？适合出门吗？"},
    {"role": "assistant", "content": "今天天气很好，阳光明媚，非常适合户外活动！"}
]}]
dirty = long_text * 3 + [{"messages": []}] + [
    {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}  # too short
]
result = clean_dataset(dirty)
ok("清洗: 去重", result["stats"]["duplicate"] >= 2)
ok("清洗: 去空", result["stats"]["empty"] >= 1)
ok("清洗: 返回清洗后数据", len(result["cleaned_data"]) > 0)

# 数据统计
stats = get_data_stats(sharegpt * 5)
ok("统计: 总量", stats["total_count"] == 5)
ok("统计: 含长度分布", "length_distribution" in stats)
ok("统计: 含语言检测", "language_hint" in stats)

# 数据集切分
split = split_dataset(sharegpt * 10, train_ratio=0.8)
ok("切分: 总量正确", split["train_count"] + split["val_count"] == 10)
ok("切分: 比例正确", split["train_count"] == 8)

# 预览
preview = preview_data(sharegpt * 5, 3)
ok("预览: 数量正确", len(preview) == 3)

# 内置库
ok("内置数据集", len(BUILTIN_DATASETS) >= 8, f"{len(BUILTIN_DATASETS)}个")
cats = set(d["category"] for d in BUILTIN_DATASETS)
ok("内置库含多分类", len(cats) >= 3, f"分类: {cats}")
ok("内置库含必要字段", all(all(k in d for k in ["id", "name", "format", "record_count", "license"]) for d in BUILTIN_DATASETS))

# 保存/加载
tmp_data = sharegpt * 20
saved_path = save_dataset(tmp_data, "test_dataset", "sharegpt")
ok("保存数据集", os.path.exists(saved_path))
loaded = load_dataset("test_dataset")
ok("加载数据集", loaded["success"] and loaded["record_count"] == 20)
local_ds = list_local_datasets()
ok("本地数据集列表", any(d["id"] == "test_dataset" for d in local_ds))

# 清理
from backend.dataset_manager import delete_local_dataset
delete_local_dataset("test_dataset")


# ============================================================
# Test 4: 训练引擎
# ============================================================
section("4️⃣  训练引擎 (training_engine)")

from backend.training_engine import (
    TrainingConfig, PRESET_SCHEMES,
    get_smart_recommendations, get_preset_scheme,
    detect_chat_template, CHAT_TEMPLATE_MAP,
    create_training_config, validate_training_config,
)

# 预设方案
ok("预设: quick", get_preset_scheme("quick")["num_train_epochs"] == 1)
ok("预设: standard", get_preset_scheme("standard")["lora_rank"] == 16)
ok("预设: deep", get_preset_scheme("deep")["num_train_epochs"] == 5)
ok("预设数量", len(PRESET_SCHEMES) == 3)

# 智能推荐
recs = get_smart_recommendations(7.0, 11.6, 10000)
for k in ["finetuning_type", "lora_rank", "lora_alpha", "learning_rate", "num_train_epochs",
          "per_device_train_batch_size", "gradient_accumulation_steps", "max_seq_length"]:
    ok(f"推荐含{k}", k in recs)

# 不同硬件推荐不同方案
recs_small = get_smart_recommendations(7.0, 6.0, 1000)
recs_large = get_smart_recommendations(7.0, 24.0, 1000)
ok("小显存推荐小batch", recs_small["per_device_train_batch_size"] <= recs_large["per_device_train_batch_size"])

# Chat Template
for model_id, expected in [
    ("Qwen/Qwen2.5-7B-Instruct", "chatml"),
    ("meta-llama/Llama-3.1-8B", "llama3"),
    ("mistralai/Mistral-7B-v0.1", "mistral"),
    ("google/gemma-2-2b-it", "gemma"),
    ("deepseek-ai/DeepSeek-Coder-6.7B", "deepseek"),
]:
    r = detect_chat_template(model_id)
    ok(f"ChatTemplate: {model_id.split('/')[-1][:20]}", r["template_name"] == expected,
       f"{r['template_name']} (confidence:{r['confidence']})")

ok("ChatTemplate映射表", len(CHAT_TEMPLATE_MAP) >= 10, f"{len(CHAT_TEMPLATE_MAP)}个模型族")

# 创建训练配置
config = create_training_config(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    model_path="/tmp/test-model",
    dataset_path="/tmp/data.json",
    dataset_format="sharegpt",
    finetuning_type="lora",
    preset="standard",
    available_vram_gb=11.6,
    model_params_b=7.0,
    dataset_size=10000,
)
ok("配置创建", isinstance(config, TrainingConfig))
ok("配置含VRAM预算", "total_estimate_gb" in config.vram_budget)
ok("配置含ChatTemplate", config.chat_template == "chatml")
ok("配置序列化(to_dict)", isinstance(config.to_dict(), dict))
ok("配置反序列化(from_dict)", isinstance(TrainingConfig.from_dict(config.to_dict()), TrainingConfig))

# 配置验证
validation = validate_training_config(config)
ok("配置验证返回", "valid" in validation)
ok("配置验证含警告列表", isinstance(validation.get("warnings"), list))

# 有效配置验证（使用真实模型路径）
real_model = Path("data/models/Qwen--Qwen2.5-1.5B-Instruct")
real_data = Path("data/datasets/demo-zh-conversation/data.json")
if real_model.exists() and real_data.exists():
    real_config = create_training_config(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_path=str(real_model),
        dataset_path=str(real_data),
        finetuning_type="lora",
        preset="standard",
        available_vram_gb=11.6,
        model_params_b=1.5,
        dataset_size=500,
    )
    real_val = validate_training_config(real_config)
    ok("真实路径配置有效", real_val["valid"], f"issues:{len(real_val['issues'])} warnings:{len(real_val['warnings'])}")
else:
    warn("真实模型/数据路径不可用，跳过真实配置验证")


# ============================================================
# Test 5: 训练监控
# ============================================================
section("5️⃣  训练监控 (training_monitor)")

from backend.training_monitor import (
    GPUStatus, TrainingMetrics, TrainingProgress,
    get_gpu_status, parse_training_log,
    get_training_progress, list_checkpoints,
    format_duration, format_vram_bar, get_sample_outputs,
)

# GPU状态
gpus = get_gpu_status()
if gpus:
    ok("GPU状态获取", len(gpus) > 0)
    ok("GPU含名称", len(gpus[0].name) > 0)
    ok("GPU含显存信息", gpus[0].memory_total_gb > 0)
else:
    warn("无GPU信息（可能没有nvidia-ml-py或非NVIDIA GPU）")

# 日志解析测试
sample_log = """{"step": 10, "epoch": 0.1, "loss": 2.5, "learning_rate": 0.0002}
{"step": 20, "epoch": 0.2, "loss": 2.3, "learning_rate": 0.0002}
{"step": 30, "epoch": 0.3, "loss": 2.1, "learning_rate": 0.0002}"""

# Write temp log and parse
tmp_log = Path("/tmp/test_training_log.jsonl")
tmp_log.write_text(sample_log)
metrics = parse_training_log(str(tmp_log))
ok("日志解析", len(metrics) == 3)
ok("解析含loss", all(m.loss > 0 for m in metrics))
ok("解析含step", all(m.step > 0 for m in metrics))
tmp_log.unlink()

# 格式化
ok("时长: 3723s", format_duration(3723) == "1 小时 2 分")
ok("时长: 125s", format_duration(125) == "2 分 5 秒")
ok("时长: 30s", format_duration(30) == "30 秒")
ok("显存条: 正常", "█" in format_vram_bar(6, 12) and "░" in format_vram_bar(6, 12))

# Checkpoint列表（真实训练产出）
real_exp = Path("data/experiments/demo-qwen1.5b-qlora")
if real_exp.exists():
    ckpts = list_checkpoints(str(real_exp))
    ok(f"Checkpoint列表: {len(ckpts)}个", len(ckpts) >= 1, f"含step={ckpts[0].get('step','?')} 大小={ckpts[0].get('size_mb','?')}MB")

    progress = get_training_progress(str(real_exp))
    ok("训练进度查询", progress.experiment_id == "demo-qwen1.5b-qlora")
else:
    warn("训练实验目录不存在，跳过监控测试")


# ============================================================
# Test 6: 导出模块
# ============================================================
section("6️⃣  导出模块 (export)")

from backend.export import (
    QUANTIZATION_OPTIONS, ExportResult,
    get_quantization_options, generate_model_card,
    generate_ollama_modelfile,
)

# 量化选项
opts = get_quantization_options(14.5)
ok("量化选项数量", len(opts) == 5)
ok("Q4_K_M是推荐", opts[1]["recommended"] == True)
# q4_0 is smallest, f16 is largest (not monotonic because q4_0 < q4_K_M < q5_K_M etc.)
ok("f16最大", opts[-1]["estimated_size_gb"] >= opts[0]["estimated_size_gb"], f"f16={opts[-1]['estimated_size_gb']:.1f} vs q4_0={opts[0]['estimated_size_gb']:.1f}")
ok("预估大小范围合理", all(0 < o["estimated_size_gb"] <= 15 for o in opts))

# Model Card
card = generate_model_card(
    model_name="Test-Model",
    base_model="Qwen2.5-7B-Instruct",
    finetuning_type="lora",
    dataset_name="Test-Data", dataset_size=500,
    gguf_filename="test.Q4_K_M.gguf", ollama_name="test-model",
    final_loss=1.24, training_duration="3小时12分",
    quantization="q4_K_M", description="测试模型",
)
ok("ModelCard生成", len(card) > 300)
ok("Card含基础模型", "Qwen2.5-7B-Instruct" in card)
ok("Card含Ollama命令", "ollama create" in card)
ok("Card含GGUF文件名", "test.Q4_K_M.gguf" in card)
ok("Card含参数信息", "Learning Rate" in card and "LoRA Rank" in card)

# Ollama Modelfile
mf = generate_ollama_modelfile(
    gguf_path="./test.Q4_K_M.gguf",
    model_name="test-model",
    chat_template="<|im_start|>user\n{{ .Prompt }}<|im_end|>",
    system_prompt="你是一个测试助手。",
)
ok("Modelfile含FROM", "FROM ./test.Q4_K_M.gguf" in mf)
ok("Modelfile含TEMPLATE", "TEMPLATE" in mf)
ok("Modelfile含SYSTEM", "SYSTEM" in mf)
ok("Modelfile含PARAMETER", "PARAMETER temperature" in mf)


# ============================================================
# Test 7: 实验管理
# ============================================================
section("7️⃣  实验管理 (experiment_store)")

from backend.experiment_store import (
    get_connection, create_experiment, update_experiment_status,
    get_experiment, list_experiments, delete_experiment,
    log_metric, get_metrics, compare_experiments, get_statistics,
)

# 清理旧测试数据
conn = get_connection()
conn.execute("DELETE FROM experiments WHERE id LIKE 'test_feat_%'")
conn.execute("DELETE FROM training_metrics WHERE experiment_id LIKE 'test_feat_%'")
conn.commit()
conn.close()

# CRUD
r = create_experiment("test_feat_001", "test/model", model_name="TestModel",
                      dataset_name="TestData", dataset_size=1000,
                      finetuning_type="lora", preset="standard",
                      config={"lora_rank": 16}, tags=["test"])
ok("创建实验", r["success"])

r = update_experiment_status("test_feat_001", "completed", final_loss=1.5, best_loss=1.3, training_duration_seconds=3600)
ok("更新状态", r["success"])

exp = get_experiment("test_feat_001")
ok("获取实验", exp is not None and exp["status"] == "completed")
ok("Final loss正确", exp["final_loss"] == 1.5)

# 第二个实验
create_experiment("test_feat_002", "test/model2", model_name="TestModel2",
                  dataset_name="TestData2", finetuning_type="qlora")
update_experiment_status("test_feat_002", "completed", final_loss=1.8)

# 列表
exps = list_experiments()
ok("列表查询", len(exps) >= 2)

# 对比
comp = compare_experiments(["test_feat_001", "test_feat_002"])
ok("实验对比", len(comp["experiments"]) == 2)
ok("最佳Loss选出", comp["best_loss_exp"] == "test_feat_001")

# 指标日志
log_metric("test_feat_001", 10, 2.5, epoch=0.1, learning_rate=2e-4)
log_metric("test_feat_001", 20, 2.3, epoch=0.2, learning_rate=2e-4)
metrics = get_metrics("test_feat_001")
ok("指标记录", len(metrics) == 2)
ok("指标含loss", all("loss" in m for m in metrics))

# 全局统计
stats = get_statistics()
ok("统计: 含总量", stats["total_experiments"] >= 2)
ok("统计: 含完成量", "completed_experiments" in stats)
ok("统计: 含训练时长", stats["total_training_hours"] >= 0)

# 清理
delete_experiment("test_feat_001")
delete_experiment("test_feat_002")
ok("删除实验", get_experiment("test_feat_001") is None)


# ============================================================
# Test 8: 模型卡片
# ============================================================
section("8️⃣  模型适配卡片 (model_cards)")

from backend.model_cards import (
    load_cards, match_card, get_card_by_family,
    search_cards, get_training_recommendation, get_card_display,
    list_all_families,
)

cards = load_cards()
ok("卡片库加载", len(cards) >= 10, f"{len(cards)}张卡片")
ok("卡片含必要字段", all(
    all(k in c for k in ["id", "patterns", "display_name", "family", "chat_template", "training", "notes"])
    for c in cards
))
ok("卡片training含三种方式", all(
    all(m in c["training"] for m in ["qlora", "lora", "full"])
    for c in cards
))

# 匹配
for mid, expected_name in [
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
    ("deepseek-ai/DeepSeek-Coder-6.7B-Instruct", "DeepSeek-Coder-6.7B-Instruct"),
    ("Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct"),
    ("chatglm3-6b", "ChatGLM3-6B"),
]:
    c = match_card(mid)
    if c:
        ok(f"匹配: {mid}", c["display_name"] == expected_name)
    else:
        ok(f"匹配: {mid}", False, "未匹配到卡片")

# 未匹配
ok("未匹配返回None", match_card("completely/unknown-model-v999") is None)

# family匹配
card = get_card_by_family("qwen2")
ok("Family匹配: qwen2", card is not None and "Qwen" in card["display_name"])
card = get_card_by_family("deepseek")
ok("Family匹配: deepseek", card is not None and "DeepSeek" in card["display_name"])

# 搜索
results = search_cards(tag="中文")
ok("标签搜索: 中文", len(results) >= 4, f"{len(results)}个模型")
results = search_cards(tag="代码")
ok("标签搜索: 代码", len(results) >= 1)
results = search_cards(query="7B", min_params=6, max_params=8)
ok("参数范围搜索: 6-8B", len(results) >= 2)
results = search_cards(query="llama")
ok("关键词搜索: llama", len(results) >= 1)

# 训练推荐
rec = get_training_recommendation("Qwen/Qwen2.5-7B-Instruct", 11.6)
ok("推荐: 7B+12GB显存", rec["recommended_method"] == "qlora")
ok("推荐含config", rec["config"] is not None)
ok("推荐含notes", len(rec.get("notes", [])) > 0)

rec_insuff = get_training_recommendation("Qwen/Qwen2.5-32B-Instruct", 4.0)
ok("推荐: 32B+4GB显存=不足", rec_insuff["recommended_method"] == "insufficient")

rec_unknown = get_training_recommendation("unknown/model", 16.0)
ok("推荐: 未知模型", rec_unknown["card"] is None)

# 卡片展示
display = get_card_display(cards[0])
ok("卡片展示生成", len(display) > 50)

# 族系列表
families = list_all_families()
ok("族系列表", len(families) >= 5, f"{families}")


# ============================================================
# Test 9: 前端构建
# ============================================================
section("9️⃣  前端构建 (Gradio app)")

try:
    from frontend.app import create_app, CUSTOM_CSS, AppState
    app = create_app()
    ok("Gradio Blocks创建", app is not None)
    ok("CUSTOM_CSS存在", len(CUSTOM_CSS) > 100)
    ok("AppState可实例化", isinstance(AppState(), AppState))
except Exception as e:
    ok("前端构建", False, str(e))


# ============================================================
# Test 10: GGUF量化工具
# ============================================================
section("🔟  GGUF量化工具 (quantize_gguf)")

# Check the quantize script is valid Python
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("quantize_gguf", "tools/quantize_gguf.py")
    mod = importlib.util.module_from_spec(spec)

    # Just verify syntax — don't execute (it needs a GGUF file)
    with open("tools/quantize_gguf.py") as f:
        compile(f.read(), "tools/quantize_gguf.py", "exec")
    ok("量化脚本语法正确", True)

    # Check it has the expected functions
    from tools.quantize_gguf import quantize, FTYPE_MAP
    ok("量化含FTYPE_MAP", len(FTYPE_MAP) >= 8, f"{len(FTYPE_MAP)}种量化类型")
    ok("Q4_K_M在映射中", "q4_K_M" in FTYPE_MAP)
except Exception as e:
    ok("量化模块加载", False, str(e))

# Check actual GGUF files exist
gguf_f16 = Path("data/exports/demo-qwen1.5b-qlora/demo-qwen1.5b-qlora-f16.gguf")
gguf_q4 = Path("data/exports/demo-qwen1.5b-qlora/demo-qwen1.5b-qlora-Q4_K_M.gguf")
ok("FP16 GGUF存在", gguf_f16.exists(), f"{gguf_f16.stat().st_size/(1024**3):.1f}GB" if gguf_f16.exists() else "文件缺失")
ok("Q4_K_M GGUF存在", gguf_q4.exists(), f"{gguf_q4.stat().st_size/(1024**3):.2f}GB" if gguf_q4.exists() else "文件缺失")

# Check Ollama model
ollama_model_dir = Path("data/exports/ollama-models")
if ollama_model_dir.exists():
    blob_count = len(list(ollama_model_dir.rglob("*"))) - len(list(ollama_model_dir.rglob("*/")))
    ok("Ollama模型存储", blob_count > 0, f"{blob_count}个文件/目录")
else:
    warn("Ollama模型目录不存在")


# ============================================================
# Summary
# ============================================================
section("📊 测试汇总")

total = PASS + FAIL + WARN
print(f"  通过: {PASS}/{total}")
print(f"  失败: {FAIL}/{total}")
print(f"  警告: {WARN}/{total}")

if FAIL == 0:
    print()
    print("  🎉 所有功能测试通过！模型训练室 v0.2 各模块工作正常。")
    sys.exit(0)
else:
    print()
    print(f"  ⚠️  {FAIL} 个测试失败，需要修复。")
    sys.exit(1)
