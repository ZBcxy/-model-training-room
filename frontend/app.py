"""
模型训练室 - Gradio 前端界面

三栏布局 + 六页导航 + 实时状态栏
"""

import json
import os
import sys
import time
from pathlib import Path

import gradio as gr

# Add parent dir to path so we can import backend
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hardware_checker import (
    SystemInfo,
    check_environment_deps,
    check_training_feasibility,
    generate_env_report,
    get_system_info,
    recommend_method,
)
from backend.model_hub import (
    MODEL_RECOMMENDATIONS,
    ModelDownloader,
    classify_license,
    delete_local_model,
    get_disk_usage_for_models,
    get_model_detail,
    list_local_models,
    unified_search,
)
from backend.dataset_manager import (
    BUILTIN_DATASETS,
    DatasetInfo,
    clean_dataset,
    convert_to_sharegpt,
    detect_format,
    get_data_stats,
    import_file,
    list_local_datasets,
    preview_data,
    save_dataset,
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
    TrainingProgress,
    format_duration,
    get_gpu_status,
    get_training_progress,
    list_checkpoints,
)
from backend.export import (
    QUANTIZATION_OPTIONS,
    ExportResult,
    export_to_gguf,
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
from backend.model_cards import (
    get_training_recommendation,
    match_card,
    search_cards,
    get_card_display,
    load_cards,
)


# ============================================================
# CSS Theme
# ============================================================

CUSTOM_CSS = """
:root {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --accent: #6366f1;
    --accent-hover: #818cf8;
    --success: #22c55e;
    --warning: #eab308;
    --danger: #ef4444;
    --border: #334155;
    --radius: 8px;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}

.gradio-container {
    max-width: 100% !important;
    background: var(--bg-primary) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Header */
.header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    padding: 16px 24px;
    border-radius: var(--radius);
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: var(--shadow);
}
.header h1 {
    color: white;
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
}
.header p {
    color: rgba(255,255,255,0.8);
    margin: 0;
    font-size: 0.875rem;
}

/* Status Bar */
.status-bar {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 20px;
    margin-top: 16px;
    display: flex;
    gap: 24px;
    font-size: 0.875rem;
    color: var(--text-secondary);
}
.status-bar .indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-bar .indicator.green { background: var(--success); }
.status-bar .indicator.yellow { background: var(--warning); }
.status-bar .indicator.red { background: var(--danger); }

/* Card */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 12px;
    box-shadow: var(--shadow);
}

/* Tags */
.tag-commercial { color: #22c55e; background: rgba(34, 197, 94, 0.1); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
.tag-restricted { color: #eab308; background: rgba(234, 179, 8, 0.1); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
.tag-forbidden { color: #ef4444; background: rgba(239, 68, 68, 0.1); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
.tag-unknown { color: #94a3b8; background: rgba(148, 163, 184, 0.1); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }

/* Tab styling */
.tabs > .tab-nav > button {
    font-size: 1rem !important;
    padding: 10px 20px !important;
}

footer { display: none !important; }
"""

# ============================================================
# State Management
# ============================================================

class AppState:
    """应用全局状态"""
    def __init__(self):
        self.system_info: SystemInfo | None = None
        self.current_model: dict | None = None
        self.current_dataset: list | None = None
        self.current_dataset_meta: dict = {}
        self.training_config: TrainingConfig | None = None
        self.current_experiment_id: str = ""


app_state = AppState()

# ============================================================
# Helper Functions
# ============================================================

def format_model_result_html(r) -> str:
    """将模型搜索结果格式化为 HTML 卡片"""
    lic = r.get("license_info", {})
    lic_label = lic.get("label", "⚪ 未知")
    lic_class = lic.get("class", "unknown")
    tag_class = f"tag-{lic_class}"

    comp = r.get("compatibility", {})
    comp_icon = "✅" if comp.get("supported") else "❓"

    dl_badge = "📦 已下载" if r.get("is_downloaded") else ""

    return f"""
    <div class="card" style="margin-bottom: 8px; padding: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <strong>{r['model_id']}</strong>
                <span class="{tag_class}" style="margin-left: 8px;">{lic_label}</span>
                {comp_icon}
                {f'<span style="color: #22c55e; margin-left: 8px;">{dl_badge}</span>' if dl_badge else ''}
            </div>
        </div>
        <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 4px;">
            来源: {r['source']} · 📥 {r.get('downloads', 0):,}
            {f" · 作者: {r.get('author', '')}" if r.get('author') else ''}
        </div>
    </div>
    """


# ============================================================
# Page 0: Environment Check
# ============================================================

def create_env_tab():
    """环境检测页面"""
    with gr.Column():
        gr.Markdown("## 🖥️ 环境检测")

        env_check_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")
        env_output = gr.Textbox(
            label="检测报告",
            lines=25,
            max_lines=40,
            interactive=False,
        )

        with gr.Row():
            gpu_info_box = gr.Textbox(label="GPU 信息", lines=6, interactive=False)
            vram_recommend_box = gr.Textbox(label="微调能力评估", lines=6, interactive=False)

        def run_env_check():
            report = generate_env_report()
            info = get_system_info()
            gpu_text = ""
            if info.has_gpu:
                for g in info.gpus:
                    gpu_text += f"🎮 {g.name}\n显存: {g.vram_total_gb:.1f} GB\nCUDA: {g.cuda_version}\n"
            else:
                gpu_text = "❌ 未检测到 GPU"

            rec = recommend_method(info.max_single_vram_gb)
            vram_text = f"可用显存: {info.max_single_vram_gb:.1f} GB\n推荐方式: {rec['recommended'].upper()}\n"
            for opt in rec["options"]:
                vram_text += f"{opt['icon']} {opt['label']}\n"

            return report, gpu_text, vram_text

        env_check_btn.click(
            run_env_check,
            outputs=[env_output, gpu_info_box, vram_recommend_box],
        )


# ============================================================
# Page 1: Model Hub
# ============================================================

def create_model_tab():
    """模型中心页面"""
    with gr.Column():
        gr.Markdown("## 📦 模型中心")

        # Search
        with gr.Row():
            search_input = gr.Textbox(
                label="搜索模型",
                placeholder="输入模型名称、任务、参数量...",
                scale=4,
            )
            source_check = gr.CheckboxGroup(
                choices=["huggingface", "modelscope"],
                value=["huggingface"],
                label="数据源",
                scale=1,
            )
            search_btn = gr.Button("🔍 搜索", variant="primary", scale=1)

        # Results
        search_results = gr.HTML(label="搜索结果")

        # Model detail
        with gr.Row():
            with gr.Column(scale=2):
                model_detail = gr.JSON(label="模型详情")
            with gr.Column(scale=1):
                download_action_btn = gr.Button("📥 下载模型", variant="primary", size="lg")
                download_progress = gr.Progress()

        # Local models
        gr.Markdown("### 📦 本地模型库")
        local_models_list = gr.Dataframe(
            headers=["模型ID", "大小(GB)", "架构", "许可证", "路径"],
            datatype=["str", "number", "str", "str", "str"],
            interactive=False,
            label="已下载的模型",
        )
        refresh_local_btn = gr.Button("🔄 刷新")

        # Smart Recommendations
        gr.Markdown("### 🧠 智能推荐")
        with gr.Row():
            rec_task = gr.Dropdown([("💬 对话", "chat"), ("💻 代码", "code"), ("🌐 翻译", "translation"), ("🛒 客服", "customer"), ("🔒 安全对齐", "safety")], value="chat", label="任务类型", scale=1)
            rec_lang = gr.Dropdown([("中文", "zh"), ("英文", "en"), ("中英双语", "mixed")], value="zh", label="语言偏好", scale=1)
            rec_level = gr.Dropdown([("入门", "beginner"), ("进阶", "intermediate"), ("高级", "advanced")], value="intermediate", label="用户级别", scale=1)
            rec_btn = gr.Button("🎯 为我推荐", variant="primary", scale=1)
        rec_output = gr.HTML(label="推荐结果")
        rec_detail = gr.JSON(label="详情", visible=False)

        def do_smart_recommend(task, lang, level):
            from backend.model_recommender import recommend_models
            info = get_system_info()
            vram = info.max_single_vram_gb if info.has_gpu else 8.0
            recs = recommend_models(task=task, language=lang, level=level, available_vram_gb=vram, max_results=5)
            html = f"<div style='font-size:0.9rem;'>🖥️ 检测到 {info.gpus[0].name if info.has_gpu else 'CPU'} · 显存 {vram:.1f}GB</div><br>"
            for r in recs:
                bar = "█" * int(r.score // 10) + "░" * (10 - int(r.score // 10))
                html += f"""<div class="card" style="padding:8px; margin-bottom:4px;">
                <strong>{r.rank}. {r.display_name}</strong> · {r.card['params_b']}B · {r.card['license'] if r.card else ''}<br>
                <small>[{bar}] {r.score}分 · {r.reason}</small></div>"""
            return html
        rec_btn.click(do_smart_recommend, [rec_task, rec_lang, rec_level], [rec_output])

        # Event handlers
        def do_search(query, sources):
            if not query:
                return "<div style='color: #94a3b8;'>请输入搜索关键词</div>"
            results = unified_search(query, sources=sources, limit=15)
            if not results:
                return "<div style='color: #94a3b8;'>未找到结果。请尝试其他关键词。</div>"
            html = "".join(format_model_result_html(r) for r in results)
            return html

        search_btn.click(
            do_search,
            inputs=[search_input, source_check],
            outputs=[search_results],
        )

        def refresh_local():
            models = list_local_models()
            if not models:
                return []
            return [
                [m["model_id"], m.get("size_gb", 0), str(m.get("architecture", [])), m.get("license_info", {}).get("label", ""), m.get("local_path", "")]
                for m in models
            ]

        refresh_local_btn.click(refresh_local, outputs=[local_models_list])


# ============================================================
# Page 2: Data Preparation
# ============================================================

def create_data_tab():
    """数据准备页面 — 完整工作流"""
    with gr.Column():
        gr.Markdown("## 🗂️ 数据准备")

        # Shared state for current working data
        current_data_state = gr.State(value={"data": None, "format": "", "name": "", "count": 0})

        with gr.Tabs():
            # --- Tab 1: Built-in datasets ---
            with gr.Tab("📚 内置数据集"):
                with gr.Row():
                    builtin_dropdown = gr.Dropdown(
                        choices=[f"{d['name']} ({d['record_count']:,}条 · {d['license']})" for d in BUILTIN_DATASETS],
                        label="选择数据集",
                        scale=3,
                    )
                    builtin_use_btn = gr.Button("📥 设为训练数据", variant="primary", scale=1)
                builtin_detail = gr.JSON(label="数据集详情")
                builtin_preview = gr.JSON(label="预览（前3条）")
                builtin_status = gr.Textbox(label="状态", interactive=False, lines=2)

                def show_builtin_detail(selected):
                    if not selected: return None, None, "请选择一个数据集"
                    name = selected.split(" (")[0]
                    for d in BUILTIN_DATASETS:
                        if d["name"] == name:
                            detail = {"名称": d["name"], "条数": f"{d['record_count']:,}", "大小": f"{d['size_mb']} MB", "许可证": d["license"], "格式": d["format"], "分类": d["category"], "描述": d["description"], "标签": d["tags"]}
                            return detail, d.get("sample", []), f"✅ 已选择：{d['name']}（{d['record_count']:,}条）"
                    return None, None, "未找到匹配的数据集"

                def use_builtin(selected):
                    if not selected: return "请先选择数据集", {"data": None, "count": 0}
                    name = selected.split(" (")[0]
                    for d in BUILTIN_DATASETS:
                        if d["name"] == name:
                            return f"✅ 当前训练数据：{d['name']}（{d['record_count']:,}条）", {"data": None, "format": d["format"], "name": d["name"], "count": d["record_count"]}
                    return "未找到", {"data": None, "count": 0}

                builtin_dropdown.change(show_builtin_detail, [builtin_dropdown], [builtin_detail, builtin_preview, builtin_status])
                builtin_use_btn.click(use_builtin, [builtin_dropdown], [builtin_status, current_data_state])

            # --- Tab 2: File Import ---
            with gr.Tab("✏️ 导入文件"):
                file_upload = gr.File(label="上传数据文件", file_types=[".json", ".jsonl", ".csv", ".parquet", ".txt"])
                with gr.Row():
                    import_preview_btn = gr.Button("👁 预览", scale=1)
                    import_use_btn = gr.Button("📥 设为训练数据", variant="primary", scale=1)
                import_result = gr.JSON(label="导入结果")
                import_preview = gr.JSON(label="数据预览（前5条）")
                import_status = gr.Textbox(label="状态", interactive=False, lines=2)

                def handle_upload(file):
                    if file is None: return {"错误": "请先上传文件"}, None, "未上传文件"
                    try:
                        result = import_file(file.name)
                        if result["success"]:
                            preview = preview_data(result["data"], 5)
                            return {"状态": "✅ 导入成功", "条数": result["record_count"], "检测格式": result["format"]}, preview, f"✅ 已导入 {result['record_count']} 条（{result['format']} 格式）"
                        return {"错误": result["error"]}, None, f"导入失败：{result['error']}"
                    except Exception as e:
                        return {"错误": str(e)}, None, f"导入出错：{str(e)}"

                def use_imported(file):
                    if file is None: return "请先上传文件", {"data": None, "count": 0}
                    result = import_file(file.name)
                    if result["success"]:
                        return f"✅ 当前训练数据：导入文件（{result['record_count']}条，{result['format']}）", {"data": result["data"], "format": result["format"], "name": "导入文件", "count": result["record_count"]}
                    return f"导入失败", {"data": None, "count": 0}

                file_upload.upload(handle_upload, [file_upload], [import_result, import_preview, import_status])
                import_use_btn.click(use_imported, [file_upload], [import_status, current_data_state])

            # --- Tab 3: AI Data Generation ---
            with gr.Tab("🪄 AI 生成"):
                gr.Markdown("### 写示例 → AI 批量扩充")
                gr.Markdown("写 3~5 对完美的问答示例，我们将用大模型帮你扩充到指定数量。")

                with gr.Row():
                    fewshot_count = gr.Number(label="目标生成数量", value=100, precision=0, scale=1)
                    fewshot_api = gr.Dropdown(
                        choices=["Ollama (本地)", "OpenAI (需API Key)", "Claude (需API Key)"],
                        value="Ollama (本地)",
                        label="生成引擎",
                        scale=2,
                    )
                    fewshot_api_key = gr.Textbox(label="API Key（OpenAI/Claude需要）", type="password", placeholder="sk-...", scale=2)

                fewshot_examples = gr.Textbox(
                    label="示例数据（每行一条，格式：Q: 问题 | A: 回答）",
                    lines=10,
                    placeholder="Q: 你好，今天天气怎么样？\nA: 今天天气晴朗，温度20-25度，适合户外活动。\n\nQ: 能推荐一道简单的菜吗？\nA: 番茄炒蛋是个好选择，简单易做还美味...\n\nQ: 什么是机器学习？\nA: 机器学习是人工智能的分支...",
                )
                fewshot_scenario = gr.Textbox(label="任务场景描述", value="中文对话助手，回答问题要详细、准确、友好", lines=2)
                fewshot_gen_btn = gr.Button("🪄 生成数据", variant="primary", size="lg")
                fewshot_result = gr.JSON(label="生成结果")
                fewshot_output = gr.Textbox(label="生成日志", lines=8, interactive=False)

                def do_fewshot_generate(count, api, api_key, examples, scenario):
                    if not examples.strip():
                        return None, "❌ 请至少写 1 个示例"
                    try:
                        lines = examples.strip().split("\n")
                        parsed = []
                        current_q, current_a = "", ""
                        for line in lines:
                            line = line.strip()
                            if not line: continue
                            if line.startswith("Q:") or line.startswith("问："):
                                if current_q and current_a:
                                    parsed.append({"instruction": current_q, "output": current_a})
                                current_q = line.split(":", 1)[1].strip() if ":" in line else line[2:].strip()
                                current_a = ""
                            elif line.startswith("A:") or line.startswith("答："):
                                current_a = line.split(":", 1)[1].strip() if ":" in line else line[2:].strip()
                            else:
                                if not current_a: current_q += "\n" + line
                                else: current_a += "\n" + line
                        if current_q and current_a:
                            parsed.append({"instruction": current_q, "output": current_a})

                        if not parsed:
                            return None, "❌ 无法解析示例。请使用 Q:/A: 格式或 问：/答：格式"

                        # If Ollama, try local generation
                        if "Ollama" in api:
                            return _generate_via_ollama(count, parsed, scenario)
                        else:
                            return None, f"⚠️ {api} 需要配置 API Key 后才能使用。请先在设置中配置。"

                    except Exception as e:
                        return None, f"❌ 生成失败：{str(e)}"

                fewshot_gen_btn.click(do_fewshot_generate, [fewshot_count, fewshot_api, fewshot_api_key, fewshot_examples, fewshot_scenario], [fewshot_result, fewshot_output])

            # --- Tab 4: Data Processing ---
            with gr.Tab("🔧 清洗 & 切分"):
                gr.Markdown("### 数据清洗和训练/验证集切分")
                with gr.Row():
                    clean_btn = gr.Button("🧹 一键清洗", scale=1)
                    stats_btn = gr.Button("📊 质量报告", scale=1)
                    split_btn = gr.Button("✂️ 切分训练/验证(9:1)", scale=1)
                process_output = gr.JSON(label="处理结果")
                process_save_btn = gr.Button("💾 保存处理后的数据", variant="primary")
                process_save_status = gr.Textbox(label="保存状态", interactive=False, lines=1)

                def do_clean(state):
                    if state.get("data") is None:
                        return {"错误": "请先选择或导入数据集"}
                    result = clean_dataset(state["data"])
                    state["data"] = result["cleaned_data"]
                    state["count"] = len(result["cleaned_data"])
                    return {"清洗结果": f"移除 {result['stats']['removed']} 条（重复 {result['stats']['duplicate']}，空 {result['stats']['empty']}）", "剩余": state["count"]}

                def do_stats(state):
                    if state.get("data") is None:
                        return {"错误": "请先选择或导入数据集"}
                    return get_data_stats(state["data"])

                def do_split(state):
                    if state.get("data") is None:
                        return {"错误": "请先选择或导入数据集"}
                    result = split_dataset(state["data"], train_ratio=0.9)
                    state["train"] = result["train"]
                    state["val"] = result["val"]
                    return {"训练集": result["train_count"], "验证集": result["val_count"]}

                def do_save(state):
                    if state.get("data") is None:
                        return "请先选择或导入数据集"
                    try:
                        name = state.get("name", "custom-dataset")
                        path = save_dataset(state["data"], name)
                        return f"✅ 已保存到 {path}"
                    except Exception as e:
                        return f"❌ 保存失败：{e}"

                clean_btn.click(do_clean, [current_data_state], [process_output])
                stats_btn.click(do_stats, [current_data_state], [process_output])
                split_btn.click(do_split, [current_data_state], [process_output])
                process_save_btn.click(do_save, [current_data_state], [process_save_status])

        # Current dataset indicator
        gr.Markdown("### 📋 当前训练数据集")
        current_dataset_display = gr.Textbox(label="状态", value="尚未选择数据集", lines=2, interactive=False)

        def update_indicator(state):
            if state.get("count", 0) > 0:
                return f"✅ {state.get('name', '数据集')} · {state['count']}条 · 格式：{state.get('format', '未知')}"
            return "尚未选择数据集"

        current_data_state.change(update_indicator, [current_data_state], [current_dataset_display])


def _generate_via_ollama(count, examples, scenario):
    """使用本地 Ollama 做数据扩充"""
    try:
        import subprocess, json, os
        from backend.env_config import get_ollama_bin
        ollama_bin = str(get_ollama_bin() or "ollama")
        if not os.path.exists(ollama_bin):
            return None, "❌ 未找到本地 Ollama。请先安装 Ollama。"

        prompt = f"""你是一个数据标注专家。根据以下示例，生成 {count} 条格式一致的中文对话训练数据。

场景：{scenario}

示例：
{json.dumps(examples, ensure_ascii=False, indent=2)}

要求：严格按照示例格式，答案准确详细，覆盖不同主题和难度。
直接输出 JSON 数组，每条包含 instruction 和 output 字段。"""

        result = subprocess.run(
            [ollama_bin, "run", "demo-qwen", prompt],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "OLLAMA_HOST": "http://127.0.0.1:11434"},
        )
        if result.returncode != 0:
            return None, f"❌ Ollama 调用失败：{result.stderr[:200]}"

        # Parse response - extract JSON array
        resp = result.stdout.strip()
        # Try to find JSON array in response
        import re
        match = re.search(r'\[.*\]', resp, re.DOTALL)
        if match:
            data = json.loads(match.group())
            converted = convert_to_sharegpt(data)
            return {"生成条数": len(converted), "状态": "✅ 成功"}, f"✅ 成功生成 {len(converted)} 条数据"
        return None, f"⚠️ 未能解析生成的 JSON。原始回复：{resp[:300]}"
    except Exception as e:
        return None, f"❌ 生成失败：{str(e)}"


# ============================================================
# Page 3: Training Configuration
# ============================================================

def create_training_tab():
    """训练配置页面 — 集成模型卡片 + 自动化训练启动"""
    with gr.Column():
        gr.Markdown("## 🧪 训练配置")

        # Shared config state
        config_state = gr.State(value=None)

        gr.Markdown("### ① 选择模型")
        with gr.Row():
            model_input = gr.Textbox(label="模型 ID 或名称", placeholder="输入 Qwen2.5-7B 或完整模型ID...", scale=3)
            model_detect_btn = gr.Button("🔍 智能匹配", variant="primary", scale=1)
        model_card_display = gr.Textbox(label="📋 模型适配卡片", lines=6, interactive=False, max_lines=12)
        model_match_result = gr.State(value=None)

        gr.Markdown("### ② 微调配置")
        with gr.Row():
            model_path_input = gr.Textbox(label="本地模型路径", placeholder="data/models/Qwen--Qwen2.5-1.5B-Instruct", scale=2)
            dataset_path_input = gr.Textbox(label="数据集路径", placeholder="data/datasets/demo-zh-conversation/data.json", scale=2)
        with gr.Row():
            finetuning_type = gr.Radio([("LoRA (>6GB) ⭐", "lora"), ("QLoRA (>4GB)", "qlora"), ("全参 (>40GB)", "full")], value="lora", label="微调方式")
            preset_radio = gr.Radio([("⚡ 快速(1h)", "quick"), ("🎯 标准(3h) ⭐", "standard"), ("🔬 深度(8h)", "deep"), ("🛠 自定义", "custom")], value="standard", label="预设方案")
        with gr.Row():
            dataset_size_input = gr.Number(label="数据条数", value=500, precision=0)
            vram_input = gr.Number(label="可用显存 (GB)", value=11.6)

        gr.Markdown("### ③ 参数（可调整）")
        with gr.Accordion("核心参数", open=True):
            with gr.Row():
                lora_rank = gr.Slider(4, 128, value=16, step=4, label="LoRA Rank")
                lora_alpha = gr.Slider(8, 256, value=32, step=8, label="LoRA Alpha")
                learning_rate = gr.Dropdown(["5e-5", "1e-4", "2e-4", "5e-4", "1e-3"], value="2e-4", label="Learning Rate")
            with gr.Row():
                num_epochs = gr.Slider(1, 20, value=3, step=1, label="Epochs")
                batch_size = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
                max_seq_length = gr.Slider(256, 8192, value=2048, step=256, label="Max Seq Length")

        with gr.Accordion("高级参数", open=False):
            with gr.Row():
                grad_accum = gr.Slider(1, 16, value=4, step=1, label="Gradient Accumulation")
                warmup_ratio = gr.Slider(0.0, 0.2, value=0.03, step=0.01, label="Warmup Ratio")
                lora_dropout = gr.Slider(0.0, 0.3, value=0.05, step=0.01, label="LoRA Dropout")
            with gr.Row():
                optim = gr.Dropdown(["adamw_torch", "adamw_8bit", "sgd", "adafactor"], value="adamw_torch", label="Optimizer")
                scheduler = gr.Dropdown(["cosine", "linear", "constant"], value="cosine", label="LR Scheduler")
            gradient_checkpointing = gr.Checkbox(value=True, label="Gradient Checkpointing")

        # Action buttons
        with gr.Row():
            generate_config_btn = gr.Button("🔧 生成配置", variant="secondary", size="lg", scale=1)
            start_training_btn = gr.Button("🚀 直接开始训练", variant="primary", size="lg", scale=1)

        config_output = gr.JSON(label="训练配置预览")
        training_result = gr.Textbox(label="训练状态", lines=6, interactive=False)

        # --- Handlers ---
        def detect_and_fill(model_query):
            """输入模型名 → 自动匹配卡片 → 返回推荐"""
            if not model_query:
                return "请先输入模型名称或 ID", None
            info = get_system_info()
            vram = info.max_single_vram_gb if info.has_gpu else 8.0
            rec = get_training_recommendation(model_query, vram)
            if rec["card"]:
                card = rec["card"]
                lines = [f"✅ 匹配：{card['display_name']}", f"📊 {card['params_b']}B · {card['size_gb']}GB · {card['license']}", f"🏗️ 架构：{card['family']} · Chat：{card['chat_template']}", f"🎯 推荐 {rec['recommended_method'].upper()}：rank={rec['config'].get('lora_rank', 16)}, lr={rec['config'].get('lr', 2e-4)}"]
                if rec.get("notes"):
                    for n in rec["notes"][:3]:
                        lines.append(f"💡 {n}")
                return "\n".join(lines), rec
            return f"⚠️ 未找到匹配的适配卡片，将使用默认参数。\n架构检测：{detect_chat_template(model_query)['template_name']}", None

        def do_generate_config(model_id, model_path, dataset_path, finetune, preset, ds_size, vram, rank, alpha, lr, epochs, bs, seqlen, ga, warmup, dropout, opt, sched, gc, match_result):
            if not model_id or not model_path:
                return {"error": "请填写模型 ID 和本地模型路径"}, None
            try:
                lr_val = float(lr)
            except ValueError:
                lr_val = 2e-4

            # Use card recommendations if available
            if match_result and match_result.get("config"):
                rec_config = match_result["config"]
                finetune = match_result.get("recommended_method", finetune)
                rank = rec_config.get("lora_rank", rank)
                alpha = rec_config.get("lora_alpha", alpha)
                lr_val = rec_config.get("lr", lr_val)
                bs = rec_config.get("batch_size", bs)
                seqlen = rec_config.get("max_seq_length", seqlen)

            custom = {"lora_rank": int(rank), "lora_alpha": int(alpha), "lora_dropout": float(dropout), "learning_rate": lr_val, "num_train_epochs": int(epochs), "per_device_train_batch_size": int(bs), "max_seq_length": int(seqlen), "gradient_accumulation_steps": int(ga), "warmup_ratio": float(warmup), "optim": opt, "lr_scheduler_type": sched, "gradient_checkpointing": gc}

            config = create_training_config(
                model_id=model_id, model_path=model_path, dataset_path=dataset_path or "",
                dataset_format="sharegpt", finetuning_type=finetune, preset=preset if preset != "custom" else "standard",
                custom_params=custom, available_vram_gb=float(vram), dataset_size=int(ds_size),
            )
            return config.to_dict(), config

        def do_start_training(config_dict, config_obj):
            if config_obj is None:
                return "❌ 请先生成训练配置"
            validation = validate_training_config(config_obj)
            if not validation["valid"]:
                issues = "\n".join(f"  · {i}" for i in validation["issues"])
                return f"❌ 配置验证失败：\n{issues}"

            from backend.training_engine import TrainingExecutor
            import time
            executor = TrainingExecutor(config_obj)
            result = executor.start(llama_factory_path="LLaMA-Factory")

            if result["success"]:
                exp_id = result.get("experiment_id", "unknown")
                create_experiment(
                    experiment_id=exp_id, model_id=config_obj.model_id,
                    model_name=config_obj.model_id, dataset_name=config_obj.dataset_path,
                    dataset_size=int(ds_size := 500), finetuning_type=config_obj.finetuning_type,
                    preset="custom", config=config_obj.to_dict(),
                )
                return f"✅ 训练已启动！\n实验 ID：{exp_id}\n输出目录：{result.get('output_dir', '')}\n前往「🔬 训练监控」页面查看进度。\n实验 ID：{exp_id}"
            return f"❌ 启动失败：{result['error']}"

        model_detect_btn.click(detect_and_fill, [model_input], [model_card_display, model_match_result])
        generate_config_btn.click(do_generate_config, [model_input, model_path_input, dataset_path_input, finetuning_type, preset_radio, dataset_size_input, vram_input, lora_rank, lora_alpha, learning_rate, num_epochs, batch_size, max_seq_length, grad_accum, warmup_ratio, lora_dropout, optim, scheduler, gradient_checkpointing, model_match_result], [config_output, config_state])
        start_training_btn.click(do_start_training, [config_output, config_state], [training_result])


# ============================================================
# Page 4: Training Monitor
# ============================================================

def create_monitor_tab():
    """训练监控页面"""
    with gr.Column():
        gr.Markdown("## 🔬 训练监控")

        experiment_id_input = gr.Textbox(
            label="实验 ID",
            placeholder="输入要监控的实验 ID",
        )

        with gr.Row():
            refresh_monitor_btn = gr.Button("🔄 刷新状态", variant="primary")
            auto_refresh_check = gr.Checkbox(label="自动刷新 (每10秒)", value=True)

        with gr.Row():
            with gr.Column(scale=2):
                # Loss plot placeholder
                loss_plot = gr.LinePlot(
                    x="step",
                    y="loss",
                    title="训练 Loss 曲线",
                )
            with gr.Column(scale=1):
                gpu_status_display = gr.Textbox(
                    label="🖥️ GPU 状态",
                    lines=8,
                    interactive=False,
                )
                training_info_display = gr.Textbox(
                    label="📊 训练信息",
                    lines=8,
                    interactive=False,
                )

        # Auto-refresh timer
        timer = gr.Timer(value=10, active=True)

        # Log viewer
        log_viewer = gr.Textbox(
            label="训练日志（最近 50 行）",
            lines=10,
            max_lines=30,
            interactive=False,
        )

        # Checkpoints
        checkpoint_list = gr.Dataframe(
            headers=["Checkpoint", "Step", "大小(MB)"],
            datatype=["str", "number", "number"],
            label="💾 Checkpoints",
            interactive=False,
        )

        def do_refresh_monitor(exp_id):
            if not exp_id:
                return None, "请输入实验 ID", "请输入实验 ID", "", []

            exp_dir = Path("data/experiments") / exp_id
            if not exp_dir.exists():
                return None, "实验目录不存在", "", "", []

            progress = get_training_progress(str(exp_dir))

            # Build plot data
            plot_data = None
            if progress.metrics_history:
                plot_data = {
                    "step": [m["step"] for m in progress.metrics_history],
                    "loss": [m["loss"] for m in progress.metrics_history],
                }

            # GPU status text
            gpu_text = ""
            for gpu in progress.gpu_status:
                gpu_text += f"GPU {gpu.index}: {gpu.name}\n"
                gpu_text += f"  显存: {gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.1f} GB\n"
                gpu_text += f"  利用率: {gpu.utilization_pct}% · 温度: {gpu.temperature_c}°C\n\n"

            # Training info
            info_text = f"状态: {progress.status}\n"
            info_text += f"Step: {progress.current_step}/{progress.total_steps}\n"
            info_text += f"Loss: {progress.current_loss:.4f} (最佳: {progress.best_loss:.4f})\n"
            if progress.elapsed_seconds > 0:
                info_text += f"已运行: {format_duration(progress.elapsed_seconds)}\n"
            if progress.estimated_remaining_seconds > 0:
                info_text += f"预计剩余: {format_duration(progress.estimated_remaining_seconds)}\n"

            # Checkpoints
            ckpts = list_checkpoints(str(exp_dir))
            ckpt_data = [[c["path"], c["step"], c["size_mb"]] for c in ckpts]

            return plot_data, gpu_text, info_text, progress.log_tail, ckpt_data

        def auto_refresh_if_enabled(exp_id, auto_enabled):
            if not auto_enabled or not exp_id:
                return None, "自动刷新已关闭", "自动刷新已关闭", "", []
            return do_refresh_monitor(exp_id)

        # Manual refresh
        refresh_monitor_btn.click(
            do_refresh_monitor,
            inputs=[experiment_id_input],
            outputs=[loss_plot, gpu_status_display, training_info_display, log_viewer, checkpoint_list],
        )

        # Auto-refresh via timer
        timer.tick(
            auto_refresh_if_enabled,
            inputs=[experiment_id_input, auto_refresh_check],
            outputs=[loss_plot, gpu_status_display, training_info_display, log_viewer, checkpoint_list],
        )


# ============================================================
# Page 5: Evaluation & Export
# ============================================================

def create_export_tab():
    """评估与导出页面"""
    with gr.Column():
        gr.Markdown("## 🏆 评估与导出")

        with gr.Row():
            # GGUF Export
            with gr.Column(scale=1):
                gr.Markdown("### 📦 GGUF 导出")
                export_model_path = gr.Textbox(
                    label="模型路径",
                    placeholder="data/experiments/exp_xxx/checkpoint-xxx",
                )
                export_name = gr.Textbox(
                    label="导出文件名",
                    placeholder="my-finetuned-model",
                )
                quantization_selector = gr.Radio(
                    choices=[
                        ("Q4_K_M (推荐 ⭐)", "q4_K_M"),
                        ("Q4_0 (最快)", "q4_0"),
                        ("Q5_K_M (更好)", "q5_K_M"),
                        ("Q8_0 (很好)", "q8_0"),
                        ("FP16 (无损)", "f16"),
                    ],
                    value="q4_K_M",
                    label="量化级别",
                )
                export_btn = gr.Button("📦 导出 GGUF", variant="primary")
                export_result = gr.Textbox(label="导出结果", lines=6, interactive=False)

                def do_export(model_path, name, quant):
                    if not model_path or not name:
                        return "请填写模型路径和导出名称"
                    result = export_to_gguf(
                        model_path=model_path,
                        output_name=name,
                        quantization=quant,
                    )
                    if result.success:
                        return f"✅ 导出成功！\n文件: {result.output_path}\n大小: {result.output_size_gb} GB\n耗时: {format_duration(result.duration_seconds)}"
                    return f"❌ 导出失败:\n{result.error}"

                export_btn.click(
                    do_export,
                    inputs=[export_model_path, export_name, quantization_selector],
                    outputs=[export_result],
                )

            # Model Card
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Model Card")
                card_model_name = gr.Textbox(label="模型名称", placeholder="My-FineTuned-Model")
                card_base_model = gr.Textbox(label="基础模型", placeholder="Qwen2.5-7B-Instruct")
                card_description = gr.Textbox(label="描述", lines=3, placeholder="这个模型是...")
                generate_card_btn = gr.Button("📝 生成 Model Card")
                card_output = gr.Textbox(label="Model Card 预览", lines=15, interactive=False)

                def do_generate_card(name, base, desc):
                    if not name or not base:
                        return "请填写模型名称和基础模型"
                    return generate_model_card(
                        model_name=name,
                        base_model=base,
                        description=desc,
                        gguf_filename=f"{name}.Q4_K_M.gguf",
                        ollama_name=name.lower(),
                    )

                generate_card_btn.click(
                    do_generate_card,
                    inputs=[card_model_name, card_base_model, card_description],
                    outputs=[card_output],
                )


# ============================================================
# Page 6: Experiment Management
# ============================================================

def create_experiment_tab():
    """实验管理页面"""
    with gr.Column():
        gr.Markdown("## 📊 实验管理")

        refresh_experiments_btn = gr.Button("🔄 刷新列表", variant="primary")

        with gr.Row():
            experiment_table = gr.Dataframe(
                headers=["实验ID", "模型", "数据", "方式", "Loss", "状态", "时间"],
                datatype=["str", "str", "str", "str", "number", "str", "str"],
                label="训练历史",
                interactive=False,
            )

        with gr.Row():
            with gr.Column():
                select_experiment_id = gr.Textbox(label="实验 ID", placeholder="exp_xxx")
                exp_detail_btn = gr.Button("查看详情")
                exp_delete_btn = gr.Button("🗑 删除", variant="stop")
                experiment_detail = gr.JSON(label="实验详情")

        stats_box = gr.JSON(label="全局统计")

        def do_refresh_experiments():
            exps = list_experiments(limit=50)
            if not exps:
                return [], {}

            rows = []
            for e in exps:
                duration = format_duration(e.get("training_duration_seconds", 0) or 0)
                rows.append([
                    e["id"],
                    e.get("model_name", e.get("model_id", "")),
                    e.get("dataset_name", ""),
                    e.get("finetuning_type", ""),
                    round(e.get("final_loss", 0) or 0, 4),
                    e.get("status", ""),
                    duration,
                ])

            stats = get_statistics()
            return rows, stats

        refresh_experiments_btn.click(
            do_refresh_experiments,
            outputs=[experiment_table, stats_box],
        )

        def do_get_detail(exp_id):
            if not exp_id:
                return {"error": "请输入实验 ID"}
            exp = get_experiment(exp_id)
            if exp is None:
                return {"error": "实验不存在"}
            return exp

        exp_detail_btn.click(do_get_detail, inputs=[select_experiment_id], outputs=[experiment_detail])

        def do_delete(exp_id):
            if not exp_id:
                return {"success": False, "error": "请输入实验 ID"}
            return delete_experiment(exp_id)

        exp_delete_btn.click(do_delete, inputs=[select_experiment_id], outputs=[experiment_detail])

        # Initial load
        refresh_experiments_btn.click(
            do_refresh_experiments,
            outputs=[experiment_table, stats_box],
        )


# ============================================================
# Main App
# ============================================================

def create_app():
    """构建完整的 Gradio 应用"""

    with gr.Blocks(
        title="模型训练室 Model Training Room",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    ) as app:
        # Header
        gr.HTML("""
        <div class="header">
            <div>
                <h1>🏠 模型训练室 Model Training Room</h1>
                <p>下载 · 准备数据 · 微调 · 导出 — 一站式模型微调工具</p>
            </div>
        </div>
        """)

        # Main tabs
        with gr.Tabs(elem_classes=["tabs"]) as tabs:
            with gr.Tab("🖥️ 环境检测"):
                create_env_tab()
            with gr.Tab("📦 模型中心"):
                create_model_tab()
            with gr.Tab("🗂️ 数据准备"):
                create_data_tab()
            with gr.Tab("🧪 训练配置"):
                create_training_tab()
            with gr.Tab("🔬 训练监控"):
                create_monitor_tab()
            with gr.Tab("🏆 评估导出"):
                create_export_tab()
            with gr.Tab("📊 实验管理"):
                create_experiment_tab()

        # Status bar
        gr.HTML("""
        <div class="status-bar" id="status-bar">
            <span><span class="indicator green"></span> GPU: 检测中...</span>
            <span>显存: --</span>
            <span>磁盘: --</span>
            <span>运行中任务: 0</span>
        </div>
        """)

        # Auto-detect system info on load
        def initial_env_check():
            info = get_system_info()
            gpu_str = f"GPU: {info.gpus[0].name}" if info.has_gpu else "GPU: ❌"
            vram_str = f"显存: {info.max_single_vram_gb:.1f}GB" if info.has_gpu else "显存: N/A"
            disk_str = f"磁盘: {info.disk_free_gb:.1f}/{info.disk_total_gb:.1f}GB"
            return f"""
            <div class="status-bar">
                <span><span class="indicator {'green' if info.has_gpu else 'red'}"></span> {gpu_str}</span>
                <span>💾 {vram_str}</span>
                <span>💿 {disk_str}</span>
                <span>⚙️ 运行中任务: 0</span>
            </div>
            """

        return app


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    import gradio as gr
    app = create_app()

    # Gradio version-adaptive launch
    gv = tuple(int(x) for x in gr.__version__.split(".")[:2])
    launch_kwargs = dict(server_name="127.0.0.1", server_port=7860, share=False)
    if gv >= (6, 0):
        launch_kwargs["theme"] = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate")
        launch_kwargs["css"] = CUSTOM_CSS

    print(f"🚀 模型训练室启动中... Gradio {gr.__version__}")
    print(f"   打开浏览器访问: http://127.0.0.1:7860")
    app.launch(**launch_kwargs)
