"""
模型训练室 — 简约前端
4 页面：模型 | 数据 | 训练 | 结果
"""

import json
import os
import time
from pathlib import Path

import gradio as gr

from backend.hardware_checker import get_system_info, recommend_method
from backend.model_hub import list_local_models, enhanced_search
from backend.dataset_manager import (
    BUILTIN_DATASETS, detect_format, convert_to_sharegpt,
    clean_dataset, get_data_stats, preview_data, split_dataset,
    save_dataset, import_file,
)
from backend.training_engine import (
    create_training_config, validate_training_config,
    get_smart_recommendations, detect_chat_template,
)
from backend.training_monitor import get_gpu_status, get_training_progress, format_duration, list_checkpoints
from backend.export import get_quantization_options, generate_model_card
from backend.experiment_store import create_experiment, update_experiment_status, list_experiments, get_statistics
from backend.model_cards import get_training_recommendation
from backend.model_recommender import recommend_models
from backend.env_config import get_ollama_bin, is_ollama_available

# ── 极简样式 ──────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 960px !important; margin: 0 auto !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important; }
footer { display: none !important; }
.header { padding: 16px 0 8px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 12px; }
.header h2 { margin: 0; font-weight: 600; color: #111827; }
.hw-badge { font-size: 0.75rem; color: #6b7280; padding: 2px 0; }
.section-title { font-size: 0.9rem; font-weight: 600; color: #374151; margin: 16px 0 8px 0; padding-bottom: 4px; border-bottom: 1px solid #f3f4f6; }
.card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; margin-bottom: 8px; font-size: 0.85rem; }
.result-ok { color: #059669; } .result-err { color: #dc2626; } .result-warn { color: #d97706; }
"""

# ── 初始化状态 ─────────────────────────────────────────────
info = get_system_info()
vram = info.max_single_vram_gb if info.has_gpu else 8.0

HW_BADGE = (
    f"GPU: {info.gpus[0].name} ({vram:.1f}GB) · "
    f"磁盘: {info.disk_free_gb:.0f}GB · "
    f"CUDA: {info.cuda_version}"
) if info.has_gpu else f"CPU only · 磁盘: {info.disk_free_gb:.0f}GB"


def build_header():
    gr.HTML(f"""
    <div class="header">
        <h2>模型训练室</h2>
        <div class="hw-badge">{HW_BADGE}</div>
    </div>
    """)


# ============================================================
# Page 1: 模型
# ============================================================
def page_models():
    with gr.Column():
        # 搜索
        with gr.Row():
            q = gr.Textbox(placeholder="搜索模型...", scale=4, show_label=False, container=False)
            src = gr.CheckboxGroup(["huggingface", "modelscope"], value=["huggingface", "modelscope"], label="源", scale=1)
            btn = gr.Button("搜索", scale=1, size="sm")

        results = gr.HTML()

        # 本地模型
        gr.Markdown('<div class="section-title">本地模型</div>')
        local = gr.Dataframe(
            headers=["模型ID", "大小", "许可证"],
            datatype=["str", "str", "str"],
            interactive=False,
        )

        def do_search(query, sources):
            if not query: return "<div style='color:#9ca3af;font-size:0.85rem;'>输入关键词搜索</div>"
            r = enhanced_search(query, sources=sources, limit=10)
            if not r["results"]:
                return f"<div style='color:#9ca3af;font-size:0.85rem;'>无结果 · {r['source']}</div>"
            html = ""
            for item in r["results"]:
                card = item.get("card_match")
                card_info = ""
                if card:
                    card_info = f"<span style='color:#6b7280;'> · {card['params_b']}B · {card['family']} · QLoRA≥{card['min_vram_qlora']}GB</span>"
                lic = item.get("license_label", "")
                downloaded = "📥" if item.get("is_downloaded") else ""
                html += f"""<div class="card">{downloaded} <strong>{item['model_id']}</strong>{card_info}<br><span style="font-size:0.75rem;color:#9ca3af;">{item['source']} · {lic}</span></div>"""
            return html

        def refresh_local():
            models = list_local_models()
            return [[m["model_id"], f"{m.get('size_gb',0):.1f}GB", m.get("license_info",{}).get("label","")] for m in models]

        btn.click(do_search, [q, src], [results])
        gr.Button("刷新", size="sm").click(refresh_local, outputs=[local])


# ============================================================
# Page 2: 数据
# ============================================================
def page_data():
    state = gr.State({"data": None, "format": "", "name": "", "count": 0})

    with gr.Column():
        # 来源选择
        gr.Markdown('<div class="section-title">数据来源</div>')
        with gr.Row():
            builtin = gr.Dropdown(
                choices=[""] + [f"{d['name']} ({d['record_count']:,}条)" for d in BUILTIN_DATASETS],
                label="内置数据集", scale=2,
            )
            upload = gr.File(label="或上传文件", file_types=[".json", ".jsonl", ".csv"], scale=2)

        # AI生成
        gr.Markdown('<div class="section-title">AI 生成</div>')
        with gr.Row():
            examples_input = gr.Textbox(
                placeholder="写几个问答示例...\n问：你好\n答：你好！有什么可以帮你的？",
                lines=4, label="示例", scale=3,
            )
            gen_count = gr.Number(value=100, label="生成数量", precision=0, scale=1)
            gen_btn = gr.Button("生成", size="sm", scale=1)

        gen_log = gr.Textbox(label="生成日志", lines=2, interactive=False)

        # 处理
        gr.Markdown('<div class="section-title">处理 & 预览</div>')
        with gr.Row():
            clean_btn = gr.Button("清洗", size="sm")
            split_btn = gr.Button("切分 9:1", size="sm")
            save_btn = gr.Button("保存", size="sm", variant="primary")
        preview_box = gr.JSON(label="预览（前 5 条）")
        status_box = gr.Textbox(label="状态", interactive=False)

        def use_builtin(sel):
            if not sel: return None, "未选择", state
            name = sel.split(" (")[0]
            for d in BUILTIN_DATASETS:
                if d["name"] == name:
                    return d.get("sample", []), f"✅ {d['name']}: {d['record_count']:,}条 · {d['license']}", {"data": None, "format": d["format"], "name": d["name"], "count": d["record_count"]}
            return None, "未找到", state

        def handle_upload(file):
            if file is None: return None, "未上传", state
            r = import_file(file.name)
            if r["success"]:
                p = preview_data(r["data"], 5)
                return p, f"✅ 导入 {r['record_count']} 条 ({r['format']})", {"data": r["data"], "format": r["format"], "name": "导入文件", "count": r["record_count"]}
            return None, f"导入失败: {r['error']}", state

        def do_clean(st):
            if st["data"] is None: return None, "请先选择数据", st
            r = clean_dataset(st["data"])
            st["data"] = r["cleaned_data"]; st["count"] = len(r["cleaned_data"])
            return preview_data(st["data"], 5), f"清洗: 移除{r['stats']['removed']}条 (去重{r['stats']['duplicate']} 去空{r['stats']['empty']}) · 剩余{st['count']}条", st

        def do_split(st):
            if st["data"] is None: return None, "请先选择数据", st
            r = split_dataset(st["data"], 0.9)
            st["data"] = r["train"]  # 用训练集
            return preview_data(st["data"], 5), f"训练集 {r['train_count']} + 验证集 {r['val_count']}", st

        def do_save(st):
            if st["data"] is None: return None, "请先选择数据", st
            try:
                p = save_dataset(st["data"], st.get("name", "custom"))
                return None, f"✅ 已保存: {p}", st
            except Exception as e:
                return None, f"保存失败: {e}", st

        def do_generate(examples_str, count):
            if not examples_str.strip(): return None, "请先写几个示例"
            if not is_ollama_available(): return None, "Ollama 不可用，请先启动 Ollama"
            try:
                import subprocess
                ollama = str(get_ollama_bin())
                prompt = f"根据示例生成{count}条格式一致的对话数据。示例:\n{examples_str}\n\n直接输出JSON数组 [{{\"instruction\":\"...\",\"output\":\"...\"}}]"
                r = subprocess.run([ollama, "run", "demo-qwen", prompt], capture_output=True, text=True, timeout=180, env={**os.environ, "OLLAMA_HOST": "http://127.0.0.1:11434"})
                if r.returncode != 0: return None, f"生成失败: {r.stderr[:200]}"
                return None, f"✅ 生成完成 ({len(r.stdout)} 字符)"
            except Exception as e:
                return None, f"出错: {e}"

        builtin.change(use_builtin, [builtin], [preview_box, status_box, state])
        upload.upload(handle_upload, [upload], [preview_box, status_box, state])
        clean_btn.click(do_clean, [state], [preview_box, status_box, state])
        split_btn.click(do_split, [state], [preview_box, status_box, state])
        save_btn.click(do_save, [state], [preview_box, status_box, state])
        gen_btn.click(do_generate, [examples_input, gen_count], [preview_box, gen_log])


# ============================================================
# Page 3: 训练
# ============================================================
def page_train():
    config_state = gr.State(None)

    with gr.Column():
        gr.Markdown('<div class="section-title">模型 & 数据</div>')
        with gr.Row():
            model_id = gr.Textbox(label="模型ID", placeholder="Qwen/Qwen2.5-7B-Instruct", scale=3)
            detect_btn = gr.Button("匹配", size="sm", scale=1)
        model_path = gr.Textbox(label="本地路径", placeholder="data/models/Qwen--Qwen2.5-1.5B-Instruct")
        data_path = gr.Textbox(label="数据路径", placeholder="data/datasets/.../data.json")

        card_info = gr.Textbox(label="适配信息", lines=3, interactive=False)

        gr.Markdown('<div class="section-title">参数</div>')
        with gr.Row():
            method = gr.Radio(["lora", "qlora", "full"], value="lora", label="方式")
            preset = gr.Radio(["quick (1h)", "standard (3h)", "deep (8h)"], value="standard (3h)", label="预设")
        with gr.Row():
            lora_r = gr.Slider(4, 64, value=16, step=4, label="Rank")
            lr = gr.Dropdown(["5e-5", "1e-4", "2e-4", "5e-4"], value="2e-4", label="学习率")
            epochs = gr.Slider(1, 10, value=3, step=1, label="Epoch")
            bs = gr.Slider(1, 8, value=4, step=1, label="Batch")

        with gr.Accordion("高级", open=False):
            with gr.Row():
                seq = gr.Slider(512, 4096, value=2048, step=256, label="Max Seq")
                ga = gr.Slider(1, 8, value=4, step=1, label="Grad Acc")

        with gr.Row():
            gen_btn = gr.Button("生成配置", size="sm")
            start_btn = gr.Button("开始训练", variant="primary", size="sm")

        config_out = gr.JSON(label="配置预览")
        train_out = gr.Textbox(label="状态", lines=4, interactive=False)

        def do_detect(mid):
            if not mid: return "输入模型ID"
            rec = get_training_recommendation(mid, vram)
            if rec["card"]:
                c = rec["card"]
                return f"{c['display_name']} · {c['params_b']}B · {c['license']}\n推荐: {rec['recommended_method'].upper()} rank={rec['config'].get('lora_rank')} lr={rec['config'].get('lr')}\n{'; '.join(rec.get('notes',[])[:2])}"
            return f"未匹配 · Chat: {detect_chat_template(mid)['template_name']}"

        def do_gen(mid, mp, dp, mt, ps, rk, lrv, ep, bt, sl, ga_val):
            if not mid or not mp: return None, "请填写模型ID和路径"
            try: lr_val = float(lrv)
            except: lr_val = 2e-4
            ps_map = {"quick (1h)": "quick", "standard (3h)": "standard", "deep (8h)": "deep"}
            cfg = create_training_config(mid, mp, dp or "", "sharegpt", mt, ps_map.get(ps,"standard"), custom_params={"lora_rank":int(rk),"learning_rate":lr_val,"num_train_epochs":int(ep),"per_device_train_batch_size":int(bt),"max_seq_length":int(sl),"gradient_accumulation_steps":int(ga_val)}, available_vram_gb=vram)
            return cfg.to_dict(), cfg

        def do_start(_, cfg):
            if cfg is None: return "请先生成配置"
            v = validate_training_config(cfg)
            if not v["valid"]: return "配置错误:\n" + "\n".join(v["issues"])
            from backend.training_engine import TrainingExecutor
            e = TrainingExecutor(cfg)
            r = e.start()
            if r["success"]:
                create_experiment(r["experiment_id"], cfg.model_id, model_name=cfg.model_id, finetuning_type=cfg.finetuning_type, config=cfg.to_dict())
                return f"✅ 训练启动\nID: {r['experiment_id']}"
            return f"失败: {r['error']}"

        detect_btn.click(do_detect, [model_id], [card_info])
        gen_btn.click(do_gen, [model_id, model_path, data_path, method, preset, lora_r, lr, epochs, bs, seq, ga], [config_out, config_state])
        start_btn.click(do_start, [config_out, config_state], [train_out])


# ============================================================
# Page 4: 结果
# ============================================================
def page_results():
    with gr.Column():
        gr.Markdown('<div class="section-title">训练历史</div>')
        exp_table = gr.Dataframe(
            headers=["ID", "模型", "方式", "Loss", "状态", "时间"],
            datatype=["str", "str", "str", "number", "str", "str"],
            interactive=False,
        )

        gr.Markdown('<div class="section-title">监控</div>')
        exp_id = gr.Textbox(label="实验ID", placeholder="exp_...")
        with gr.Row():
            refresh_btn = gr.Button("刷新", size="sm")
            auto_chk = gr.Checkbox(label="自动刷新", value=True)
        timer = gr.Timer(value=10, active=True)

        with gr.Row():
            loss_plot = gr.LinePlot(x="step", y="loss", title="Loss")
            monitor_text = gr.Textbox(label="状态", lines=6, interactive=False)

        gr.Markdown('<div class="section-title">导出</div>')
        with gr.Row():
            export_path = gr.Textbox(label="模型路径", placeholder="data/exports/.../merged", scale=3)
            quant = gr.Dropdown(["q4_K_M", "q4_0", "q5_K_M", "q8_0"], value="q4_K_M", label="量化", scale=1)
            export_btn = gr.Button("导出 GGUF", size="sm", scale=1)
        export_out = gr.Textbox(label="导出状态", interactive=False)

        def refresh_exps():
            exps = list_experiments(limit=20)
            rows = []
            for e in exps:
                dur = format_duration(e.get("training_duration_seconds", 0) or 0)
                rows.append([e["id"], e.get("model_name",""), e.get("finetuning_type",""), round(e.get("final_loss",0) or 0, 4), e.get("status",""), dur])
            return rows

        def do_monitor(eid):
            if not eid: return None, "输入实验ID"
            exp_dir = Path("data/experiments") / eid
            if not exp_dir.exists(): return None, "实验不存在"
            prog = get_training_progress(str(exp_dir))
            plot = None
            if prog.metrics_history:
                plot = {"step": [m["step"] for m in prog.metrics_history], "loss": [m["loss"] for m in prog.metrics_history]}
            txt = f"Loss: {prog.current_loss:.4f} · Step: {prog.current_step}"
            if prog.elapsed_seconds > 0:
                txt += f"\n运行: {format_duration(prog.elapsed_seconds)}"
                if prog.estimated_remaining_seconds > 0:
                    txt += f" · 剩余: {format_duration(prog.estimated_remaining_seconds)}"
            return plot, txt

        def auto_monitor(eid, auto):
            if not auto or not eid: return None, ""
            return do_monitor(eid)

        def do_export(mpath, qtype):
            if not mpath: return "输入模型路径"
            try:
                from backend.export import export_to_gguf
                r = export_to_gguf(mpath, "exported", qtype)
                return f"✅ {r.output_path} ({r.output_size_gb:.1f}GB)" if r.success else f"失败: {r.error}"
            except Exception as e:
                return f"出错: {e}"

        refresh_btn.click(refresh_exps, outputs=[exp_table])
        refresh_btn.click(do_monitor, [exp_id], [loss_plot, monitor_text])
        export_btn.click(do_export, [export_path, quant], [export_out])
        timer.tick(auto_monitor, [exp_id, auto_chk], [loss_plot, monitor_text])


# ============================================================
# App
# ============================================================
def create_app():
    with gr.Blocks(title="模型训练室", css=CSS) as app:
        build_header()

        with gr.Tabs():
            with gr.Tab("模型"):
                page_models()
            with gr.Tab("数据"):
                page_data()
            with gr.Tab("训练"):
                page_train()
            with gr.Tab("结果"):
                page_results()

    return app


if __name__ == "__main__":
    app = create_app()
    gv = tuple(int(x) for x in gr.__version__.split(".")[:2])
    kw = {"server_name": "127.0.0.1", "server_port": 7860, "share": False}
    if gv >= (6, 0):
        kw["theme"] = gr.themes.Soft(primary_hue="gray", neutral_hue="gray")
        kw["css"] = CSS
    app.launch(**kw)
