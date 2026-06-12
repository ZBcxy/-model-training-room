"""
模型适配卡片库

为热门模型提供预设的训练配置、注意事项和推荐数据集。
搜索结果自动匹配卡片，用户无需手动查参数。
"""

import json
from pathlib import Path

CARDS_FILE = Path(__file__).parent.parent / "data" / "model_cards.json"


# ============================================================
# 模型适配卡片库
# ============================================================

DEFAULT_CARDS = [
    # ---- Qwen 系列 ----
    {
        "id": "qwen25-7b-instruct",
        "patterns": ["qwen2.5", "qwen2.5-7b", "qwen/qwen2.5-7b-instruct"],
        "display_name": "Qwen2.5-7B-Instruct",
        "family": "qwen2",
        "chat_template": "chatml",
        "params_b": 7.0,
        "size_gb": 14.5,
        "license": "🟢 Apache 2.0",
        "description": "阿里通义千问2.5，中文能力顶级，7B参数适合消费级显卡",
        "sources": [
            {"platform": "modelscope", "id": "Qwen/Qwen2.5-7B-Instruct"},
            {"platform": "huggingface", "id": "Qwen/Qwen2.5-7B-Instruct"},
        ],
        "training": {
            "qlora": {"min_vram_gb": 6, "lora_rank": 16, "lora_alpha": 32, "lr": 2e-4,
                      "batch_size": 2, "max_seq_length": 2048},
            "lora":  {"min_vram_gb": 16, "lora_rank": 32, "lora_alpha": 64, "lr": 1e-4,
                      "batch_size": 4, "max_seq_length": 4096},
            "full":  {"min_vram_gb": 40, "lr": 5e-5,
                      "batch_size": 2, "max_seq_length": 4096},
        },
        "notes": [
            "需要 trust_remote_code=True",
            "Chat Template 为 ChatML，自动检测",
            "中英双语，中文任务不需要额外适配",
            "支持工具调用，微调后可保留 function calling 能力",
        ],
        "recommended_datasets": ["alpaca-zh", "belle-zh", "firefly-zh", "sharegpt-zh"],
        "tags": ["中文", "对话", "指令微调", "热门"],
    },
    {
        "id": "qwen25-1.5b-instruct",
        "patterns": ["qwen2.5-1.5b", "qwen/qwen2.5-1.5b-instruct"],
        "display_name": "Qwen2.5-1.5B-Instruct",
        "family": "qwen2",
        "chat_template": "chatml",
        "params_b": 1.5,
        "size_gb": 3.0,
        "license": "🟢 Apache 2.0",
        "description": "Qwen2.5 轻量版，3GB，入门首选，适合学习和快速实验",
        "sources": [
            {"platform": "modelscope", "id": "Qwen/Qwen2.5-1.5B-Instruct"},
            {"platform": "huggingface", "id": "Qwen/Qwen2.5-1.5B-Instruct"},
        ],
        "training": {
            "qlora": {"min_vram_gb": 4, "lora_rank": 8, "lora_alpha": 16, "lr": 5e-4,
                      "batch_size": 4, "max_seq_length": 1024},
            "lora":  {"min_vram_gb": 6, "lora_rank": 16, "lora_alpha": 32, "lr": 2e-4,
                      "batch_size": 8, "max_seq_length": 2048},
            "full":  {"min_vram_gb": 10, "lr": 1e-4,
                      "batch_size": 4, "max_seq_length": 2048},
        },
        "notes": [
            "显存友好，6GB 即可 LoRA 微调",
            "适合快速验证和数据 Pipeline 测试",
            "效果不如 7B 版本，但训练速度快 4-5 倍",
        ],
        "recommended_datasets": ["alpaca-zh", "firefly-zh", "dolly-zh"],
        "tags": ["中文", "入门", "轻量", "快速实验"],
    },
    {
        "id": "qwen25-32b-instruct",
        "patterns": ["qwen2.5-32b", "qwen/qwen2.5-32b-instruct"],
        "display_name": "Qwen2.5-32B-Instruct",
        "family": "qwen2",
        "chat_template": "chatml",
        "params_b": 32.0,
        "size_gb": 65.0,
        "license": "🟢 Apache 2.0",
        "description": "Qwen2.5 大杯，能力强劲，需要高端显卡或云端 GPU",
        "sources": [
            {"platform": "modelscope", "id": "Qwen/Qwen2.5-32B-Instruct"},
            {"platform": "huggingface", "id": "Qwen/Qwen2.5-32B-Instruct"},
        ],
        "training": {
            "qlora": {"min_vram_gb": 12, "lora_rank": 32, "lora_alpha": 64, "lr": 1e-4,
                      "batch_size": 1, "max_seq_length": 1024},
            "lora":  {"min_vram_gb": 24, "lora_rank": 64, "lora_alpha": 128, "lr": 5e-5,
                      "batch_size": 2, "max_seq_length": 2048},
            "full":  {"min_vram_gb": 80, "lr": 2e-5,
                      "batch_size": 1, "max_seq_length": 2048},
        },
        "notes": [
            "QLoRA 在 12GB 卡上可跑，但 batch_size=1, max_seq<2048",
            "推荐使用 DeepSpeed ZeRO-3 做多卡训练",
            "全参微调需要 A100/H100 级别显卡",
        ],
        "recommended_datasets": ["alpaca-zh", "belle-zh", "sharegpt-zh"],
        "tags": ["中文", "大模型", "高性能", "云端"],
    },

    # ---- Llama 系列 ----
    {
        "id": "llama31-8b-instruct",
        "patterns": ["llama-3.1-8b", "llama3.1-8b", "meta-llama/llama-3.1-8b-instruct"],
        "display_name": "Llama-3.1-8B-Instruct",
        "family": "llama",
        "chat_template": "llama3",
        "params_b": 8.0,
        "size_gb": 16.0,
        "license": "🟢 Llama 3.1 Community",
        "description": "Meta 最新开源模型，多语言能力强，社区生态最成熟",
        "sources": [
            {"platform": "huggingface", "id": "meta-llama/Llama-3.1-8B-Instruct"},
        ],
        "training": {
            "qlora": {"min_vram_gb": 6, "lora_rank": 16, "lora_alpha": 32, "lr": 2e-4,
                      "batch_size": 2, "max_seq_length": 2048},
            "lora":  {"min_vram_gb": 16, "lora_rank": 32, "lora_alpha": 64, "lr": 1e-4,
                      "batch_size": 4, "max_seq_length": 4096},
            "full":  {"min_vram_gb": 42, "lr": 5e-5,
                      "batch_size": 2, "max_seq_length": 4096},
        },
        "notes": [
            "Llama3 Chat Template，自动检测",
            "中文能力不如 Qwen2.5，英文和多语言优秀",
            "社区工具链最完善，各种教程最多",
        ],
        "recommended_datasets": ["code-alpaca", "dolly-zh"],
        "tags": ["多语言", "对话", "代码", "社区成熟"],
    },

    # ---- DeepSeek 系列 ----
    {
        "id": "deepseek-coder-6.7b",
        "patterns": ["deepseek-coder", "deepseek-coder-6.7b", "deepseek-ai/deepseek-coder-6.7b-instruct"],
        "display_name": "DeepSeek-Coder-6.7B-Instruct",
        "family": "deepseek",
        "chat_template": "deepseek",
        "params_b": 6.7,
        "size_gb": 13.4,
        "license": "🟢 DeepSeek",
        "description": "深度求索代码模型，编程任务表现优异",
        "sources": [
            {"platform": "huggingface", "id": "deepseek-ai/DeepSeek-Coder-6.7B-Instruct"},
        ],
        "training": {
            "qlora": {"min_vram_gb": 6, "lora_rank": 16, "lora_alpha": 32, "lr": 2e-4,
                      "batch_size": 2, "max_seq_length": 2048},
            "lora":  {"min_vram_gb": 14, "lora_rank": 32, "lora_alpha": 64, "lr": 1e-4,
                      "batch_size": 4, "max_seq_length": 4096},
            "full":  {"min_vram_gb": 36, "lr": 5e-5,
                      "batch_size": 2, "max_seq_length": 4096},
        },
        "notes": [
            "DeepSeek Chat Template（不同于 ChatML）",
            "代码填空（FIM）能力独特，训练数据应包含代码",
            "recommended_datasets 优先选择代码数据集",
        ],
        "recommended_datasets": ["code-alpaca"],
        "tags": ["代码", "编程", "FIM"],
    },
    {
        "id": "deepseek-v3",
        "patterns": ["deepseek-v3", "deepseek-chat", "deepseek-ai/deepseek-v3"],
        "display_name": "DeepSeek-V3",
        "family": "deepseek",
        "chat_template": "deepseek",
        "params_b": 671.0,
        "size_gb": 1342.0,
        "license": "🟢 DeepSeek",
        "description": "深度求索旗舰 MoE 模型，671B 总参数（37B 激活），性能接近 GPT-4",
        "sources": [
            {"platform": "huggingface", "id": "deepseek-ai/DeepSeek-V3"},
        ],
        "training": {
            "qlora": {"min_vram_gb": 24, "lora_rank": 64, "lora_alpha": 128, "lr": 5e-5,
                      "batch_size": 1, "max_seq_length": 512},
            "lora":  {"min_vram_gb": 48, "lora_rank": 128, "lora_alpha": 256, "lr": 2e-5,
                      "batch_size": 1, "max_seq_length": 1024},
            "full":  {"min_vram_gb": 999, "lr": 1e-5,
                      "batch_size": 1, "max_seq_length": 512},
        },
        "notes": [
            "⚠️ MoE 架构，671B 参数但每token只激活37B",
            "全参微调几乎不可能，推荐 QLoRA/LoRA",
            "需要至少 24GB 显存做 QLoRA",
            "云端 GPU（A100 80GB × 2+）推荐",
        ],
        "recommended_datasets": [],
        "tags": ["大模型", "MoE", "旗舰", "云端"],
    },

    # ---- Mistral 系列 ----
    {
        "id": "mistral-7b-v0.1",
        "patterns": ["mistral-7b", "mistralai/mistral-7b-instruct", "mistral-7b-instruct-v0.1"],
        "display_name": "Mistral-7B-Instruct-v0.1",
        "family": "mistral",
        "chat_template": "mistral",
        "params_b": 7.0,
        "size_gb": 14.5,
        "license": "🟢 Apache 2.0",
        "description": "Mistral AI 出品，7B 参数中的标杆，英文对话一流",
        "sources": [
            {"platform": "huggingface", "id": "mistralai/Mistral-7B-Instruct-v0.1"},
        ],
        "training": {
            "qlora": {"min_vram_gb": 6, "lora_rank": 16, "lora_alpha": 32, "lr": 2e-4,
                      "batch_size": 2, "max_seq_length": 2048},
            "lora":  {"min_vram_gb": 16, "lora_rank": 32, "lora_alpha": 64, "lr": 1e-4,
                      "batch_size": 4, "max_seq_length": 4096},
            "full":  {"min_vram_gb": 40, "lr": 5e-5,
                      "batch_size": 2, "max_seq_length": 4096},
        },
        "notes": [
            "Mistral Chat Template：[INST]...[/INST] 格式",
            "英文任务极强，中文弱于 Qwen",
            "滑动窗口注意力，长文本效率高",
        ],
        "recommended_datasets": [],
        "tags": ["英文", "对话", "标杆"],
    },

    # ---- Gemma 系列 ----
    {
        "id": "gemma-2-2b-it",
        "patterns": ["gemma-2-2b", "gemma-2b", "google/gemma-2-2b-it"],
        "display_name": "Gemma-2-2B-Instruct",
        "family": "gemma",
        "chat_template": "gemma",
        "params_b": 2.0,
        "size_gb": 4.0,
        "license": "🟢 Gemma",
        "description": "Google 轻量模型，4GB，小显存入门首选",
        "sources": [
            {"platform": "huggingface", "id": "google/gemma-2-2b-it"},
        ],
        "training": {
            "qlora": {"min_vram_gb": 4, "lora_rank": 8, "lora_alpha": 16, "lr": 5e-4,
                      "batch_size": 4, "max_seq_length": 1024},
            "lora":  {"min_vram_gb": 8, "lora_rank": 16, "lora_alpha": 32, "lr": 2e-4,
                      "batch_size": 8, "max_seq_length": 2048},
            "full":  {"min_vram_gb": 12, "lr": 1e-4,
                      "batch_size": 4, "max_seq_length": 2048},
        },
        "notes": [
            "Gemma Chat Template（<start_of_turn> 格式）",
            "英文为主，中文能力有限",
            "适合资源受限环境（笔记本/入门卡）",
        ],
        "recommended_datasets": [],
        "tags": ["轻量", "入门", "英文"],
    },

    # ---- Yi 系列 ----
    {
        "id": "yi-1.5-6b-chat",
        "patterns": ["yi-1.5", "yi-1.5-6b", "01-ai/yi-1.5-6b-chat"],
        "display_name": "Yi-1.5-6B-Chat",
        "family": "yi",
        "chat_template": "chatml",
        "params_b": 6.0,
        "size_gb": 12.0,
        "license": "🟢 Apache 2.0",
        "description": "零一万物中英双语模型，6B 参数，中英平衡",
        "sources": [
            {"platform": "huggingface", "id": "01-ai/Yi-1.5-6B-Chat"},
        ],
        "training": {
            "qlora": {"min_vram_gb": 6, "lora_rank": 16, "lora_alpha": 32, "lr": 2e-4,
                      "batch_size": 2, "max_seq_length": 2048},
            "lora":  {"min_vram_gb": 14, "lora_rank": 32, "lora_alpha": 64, "lr": 1e-4,
                      "batch_size": 4, "max_seq_length": 4096},
            "full":  {"min_vram_gb": 32, "lr": 5e-5,
                      "batch_size": 2, "max_seq_length": 4096},
        },
        "notes": [
            "Chat Template 为 ChatML（和 Qwen 一样）",
            "中英文能力均衡",
            "零一万物出品，社区活跃",
        ],
        "recommended_datasets": ["alpaca-zh", "belle-zh"],
        "tags": ["中文", "中英双语", "平衡"],
    },

    # ---- InternLM 系列 ----
    {
        "id": "internlm2-chat-7b",
        "patterns": ["internlm2", "internlm2-chat-7b", "internlm/internlm2-chat-7b"],
        "display_name": "InternLM2-Chat-7B",
        "family": "intern2",
        "chat_template": "intern2",
        "params_b": 7.0,
        "size_gb": 14.5,
        "license": "🟢 Apache 2.0",
        "description": "上海AI实验室出品，中文能力扎实，工具调用出色",
        "sources": [
            {"platform": "modelscope", "id": "Shanghai_AI_Laboratory/internlm2-chat-7b"},
            {"platform": "huggingface", "id": "internlm/internlm2-chat-7b"},
        ],
        "training": {
            "qlora": {"min_vram_gb": 6, "lora_rank": 16, "lora_alpha": 32, "lr": 2e-4,
                      "batch_size": 2, "max_seq_length": 2048},
            "lora":  {"min_vram_gb": 16, "lora_rank": 32, "lora_alpha": 64, "lr": 1e-4,
                      "batch_size": 4, "max_seq_length": 4096},
            "full":  {"min_vram_gb": 40, "lr": 5e-5,
                      "batch_size": 2, "max_seq_length": 4096},
        },
        "notes": [
            "Intern2 Chat Template",
            "上下文长度可达 200K（长文本能力出众）",
            "工具调用/Agent 能力出色",
        ],
        "recommended_datasets": ["alpaca-zh", "firefly-zh"],
        "tags": ["中文", "长文本", "工具调用"],
    },

    # ---- ChatGLM 系列 ----
    {
        "id": "chatglm3-6b",
        "patterns": ["chatglm3", "chatglm3-6b", "thudm/chatglm3-6b"],
        "display_name": "ChatGLM3-6B",
        "family": "chatglm",
        "chat_template": "chatglm3",
        "params_b": 6.0,
        "size_gb": 12.0,
        "license": "🟢 Apache 2.0",
        "description": "清华智谱出品，中文对话流畅，工具调用能力出色",
        "sources": [
            {"platform": "modelscope", "id": "ZhipuAI/chatglm3-6b"},
            {"platform": "huggingface", "id": "THUDM/chatglm3-6b"},
        ],
        "training": {
            "qlora": {"min_vram_gb": 6, "lora_rank": 16, "lora_alpha": 32, "lr": 2e-4,
                      "batch_size": 2, "max_seq_length": 2048},
            "lora":  {"min_vram_gb": 14, "lora_rank": 32, "lora_alpha": 64, "lr": 1e-4,
                      "batch_size": 4, "max_seq_length": 4096},
            "full":  {"min_vram_gb": 32, "lr": 5e-5,
                      "batch_size": 2, "max_seq_length": 4096},
        },
        "notes": [
            "ChatGLM3 Chat Template（特殊格式，不同于ChatML）",
            "需要 trust_remote_code=True",
            "tokenizer 特殊，注意不要替换",
            "国产模型，社区支持好",
        ],
        "recommended_datasets": ["alpaca-zh", "belle-zh", "firefly-zh"],
        "tags": ["中文", "对话", "工具调用", "国产"],
    },

    # ---- Qwen2.5-Coder 系列 ----
    {
        "id": "qwen25-coder-7b",
        "patterns": ["qwen2.5-coder-7b", "qwen/qwen2.5-coder-7b-instruct"],
        "display_name": "Qwen2.5-Coder-7B-Instruct",
        "family": "qwen2", "chat_template": "chatml",
        "params_b": 7.0, "size_gb": 14.5, "license": "🟢 Apache 2.0",
        "description": "阿里代码专用模型，编程任务能力突出",
        "sources": [{"platform":"modelscope","id":"Qwen/Qwen2.5-Coder-7B-Instruct"},{"platform":"huggingface","id":"Qwen/Qwen2.5-Coder-7B-Instruct"}],
        "training": {"qlora":{"min_vram_gb":6,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":2,"max_seq_length":2048},"lora":{"min_vram_gb":16,"lora_rank":32,"lora_alpha":64,"lr":1e-4,"batch_size":4,"max_seq_length":4096},"full":{"min_vram_gb":40,"lr":5e-5,"batch_size":2,"max_seq_length":4096}},
        "notes":["代码补全和生成能力出色","支持 FIM（Fill-in-the-Middle）","ChatML 模板"] ,
        "recommended_datasets":["code-alpaca"],
        "tags":["代码","编程","FIM","中文"],
    },
    {
        "id": "qwen25-14b-instruct",
        "patterns": ["qwen2.5-14b", "qwen/qwen2.5-14b-instruct"],
        "display_name": "Qwen2.5-14B-Instruct",
        "family": "qwen2", "chat_template": "chatml",
        "params_b": 14.0, "size_gb": 29.0, "license": "🟢 Apache 2.0",
        "description": "Qwen2.5 中杯，14B参数，性能接近大模型水平",
        "sources": [{"platform":"modelscope","id":"Qwen/Qwen2.5-14B-Instruct"},{"platform":"huggingface","id":"Qwen/Qwen2.5-14B-Instruct"}],
        "training": {"qlora":{"min_vram_gb":8,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":1,"max_seq_length":1024},"lora":{"min_vram_gb":20,"lora_rank":32,"lora_alpha":64,"lr":1e-4,"batch_size":2,"max_seq_length":2048},"full":{"min_vram_gb":60,"lr":5e-5,"batch_size":1,"max_seq_length":2048}},
        "notes":["性能接近 32B 级别","中文能力顶尖","需要较高显存"],
        "recommended_datasets":["alpaca-zh","belle-zh","firefly-zh"],
        "tags":["中文","高性能","对话"],
    },

    # ---- ChatGLM4 系列 ----
    {
        "id": "chatglm4-9b",
        "patterns": ["chatglm4", "chatglm4-9b", "thudm/glm-4-9b-chat"],
        "display_name": "GLM-4-9B-Chat",
        "family": "chatglm", "chat_template": "chatglm3",
        "params_b": 9.0, "size_gb": 18.0, "license": "🟢 Apache 2.0",
        "description": "智谱 GLM-4 最新开源版，9B 参数，中文和工具调用能力出色",
        "sources": [{"platform":"modelscope","id":"ZhipuAI/glm-4-9b-chat"},{"platform":"huggingface","id":"THUDM/glm-4-9b-chat"}],
        "training": {"qlora":{"min_vram_gb":6,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":2,"max_seq_length":2048},"lora":{"min_vram_gb":16,"lora_rank":32,"lora_alpha":64,"lr":1e-4,"batch_size":4,"max_seq_length":4096},"full":{"min_vram_gb":42,"lr":5e-5,"batch_size":2,"max_seq_length":4096}},
        "notes":["需要 trust_remote_code=True","ChatGLM 特殊 tokenizer","工具调用/Agent 能力出色","支持 128K 上下文"],
        "recommended_datasets":["alpaca-zh","belle-zh"],
        "tags":["中文","工具调用","Agent","长文本"],
    },

    # ---- Baichuan 系列 ----
    {
        "id": "baichuan2-7b-chat",
        "patterns": ["baichuan2-7b", "baichuan-inc/baichuan2-7b-chat"],
        "display_name": "Baichuan2-7B-Chat",
        "family": "baichuan", "chat_template": "baichuan",
        "params_b": 7.0, "size_gb": 14.5, "license": "🟢 Baichuan",
        "description": "百川智能出品，中文理解和生成能力扎实",
        "sources": [{"platform":"modelscope","id":"baichuan-inc/Baichuan2-7B-Chat"},{"platform":"huggingface","id":"baichuan-inc/Baichuan2-7B-Chat"}],
        "training": {"qlora":{"min_vram_gb":6,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":2,"max_seq_length":2048},"lora":{"min_vram_gb":16,"lora_rank":32,"lora_alpha":64,"lr":1e-4,"batch_size":4,"max_seq_length":4096},"full":{"min_vram_gb":40,"lr":5e-5,"batch_size":2,"max_seq_length":4096}},
        "notes":["Baichuan Chat Template","需要 trust_remote_code=True","中文优化出色","百川在医疗和金融领域有专项优化"],
        "recommended_datasets":["alpaca-zh","belle-zh"],
        "tags":["中文","对话","国产"],
    },
    {
        "id": "baichuan2-13b-chat",
        "patterns": ["baichuan2-13b", "baichuan-inc/baichuan2-13b-chat"],
        "display_name": "Baichuan2-13B-Chat",
        "family": "baichuan", "chat_template": "baichuan",
        "params_b": 13.0, "size_gb": 26.0, "license": "🟢 Baichuan",
        "description": "百川 13B 版本，性能更强",
        "sources": [{"platform":"modelscope","id":"baichuan-inc/Baichuan2-13B-Chat"},{"platform":"huggingface","id":"baichuan-inc/Baichuan2-13B-Chat"}],
        "training": {"qlora":{"min_vram_gb":8,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":1,"max_seq_length":1024},"lora":{"min_vram_gb":20,"lora_rank":32,"lora_alpha":64,"lr":1e-4,"batch_size":2,"max_seq_length":2048},"full":{"min_vram_gb":55,"lr":5e-5,"batch_size":1,"max_seq_length":2048}},
        "notes":["13B 参数，性能接近大模型","中文表现优秀","需要较高显存"],
        "recommended_datasets":["alpaca-zh","belle-zh","firefly-zh"],
        "tags":["中文","高性能","国产"],
    },

    # ---- MiniCPM 系列 ----
    {
        "id": "minicpm-2b",
        "patterns": ["minicpm", "minicpm-2b", "openbmb/minicpm-2b"],
        "display_name": "MiniCPM-2B",
        "family": "minicpm", "chat_template": "chatml",
        "params_b": 2.0, "size_gb": 4.0, "license": "🟢 Apache 2.0",
        "description": "面壁智能轻量模型，2B 参数，性能接近 7B 模型",
        "sources": [{"platform":"modelscope","id":"OpenBMB/MiniCPM-2B"},{"platform":"huggingface","id":"openbmb/MiniCPM-2B"}],
        "training": {"qlora":{"min_vram_gb":4,"lora_rank":8,"lora_alpha":16,"lr":5e-4,"batch_size":4,"max_seq_length":1024},"lora":{"min_vram_gb":8,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":8,"max_seq_length":2048},"full":{"min_vram_gb":12,"lr":1e-4,"batch_size":4,"max_seq_length":2048}},
        "notes":["超轻量，性能惊人","手机端部署友好","微调速度快"],
        "recommended_datasets":["alpaca-zh","dolly-zh"],
        "tags":["轻量","入门","中文","端侧部署"],
    },

    # ---- Phi 系列 ----
    {
        "id": "phi-4-mini",
        "patterns": ["phi-4", "phi-4-mini", "microsoft/phi-4-mini"],
        "display_name": "Phi-4-Mini",
        "family": "phi", "chat_template": "phi4",
        "params_b": 3.8, "size_gb": 7.6, "license": "🟢 MIT",
        "description": "微软 Phi 系列，小模型高性能，推理能力强",
        "sources": [{"platform":"huggingface","id":"microsoft/Phi-4-mini-instruct"}],
        "training": {"qlora":{"min_vram_gb":4,"lora_rank":8,"lora_alpha":16,"lr":5e-4,"batch_size":4,"max_seq_length":1024},"lora":{"min_vram_gb":8,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":8,"max_seq_length":2048},"full":{"min_vram_gb":16,"lr":1e-4,"batch_size":4,"max_seq_length":2048}},
        "notes":["英文和推理能力极强","中文能力弱于国产模型","小尺寸高性能"],
        "recommended_datasets":[],
        "tags":["轻量","推理","英文"],
    },

    # ---- Orion 系列 ----
    {
        "id": "orion-14b-chat",
        "patterns": ["orion", "orion-14b", "orionstarai/orion-14b-chat"],
        "display_name": "Orion-14B-Chat",
        "family": "orion", "chat_template": "chatml",
        "params_b": 14.0, "size_gb": 28.0, "license": "🟢 Apache 2.0",
        "description": "猎户星空出品，中文表现优异，多语言支持",
        "sources": [{"platform":"modelscope","id":"OrionStarAI/Orion-14B-Chat"},{"platform":"huggingface","id":"OrionStarAI/Orion-14B-Chat"}],
        "training": {"qlora":{"min_vram_gb":8,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":1,"max_seq_length":1024},"lora":{"min_vram_gb":20,"lora_rank":32,"lora_alpha":64,"lr":1e-4,"batch_size":2,"max_seq_length":2048},"full":{"min_vram_gb":60,"lr":5e-5,"batch_size":1,"max_seq_length":2048}},
        "notes":["中文多任务能力强","ChatML 模板","罕见支持日语和韩语"],
        "recommended_datasets":["alpaca-zh","belle-zh"],
        "tags":["中文","多语言","高性能"],
    },

    # ---- XuanYuan 系列 ----
    {
        "id": "xuanyuan-6b",
        "patterns": ["xuanyuan", "xuanyuan-6b", "duguangxuanyuan"],
        "display_name": "XuanYuan-6B",
        "family": "baichuan", "chat_template": "chatml",
        "params_b": 6.0, "size_gb": 12.0, "license": "🟢 Apache 2.0",
        "description": "度小满金融大模型，金融领域专项优化",
        "sources": [{"platform":"modelscope","id":"Duxiaoman-DI/XuanYuan-6B"},{"platform":"huggingface","id":"xyz-nlp/XuanYuan2.0"}],
        "training": {"qlora":{"min_vram_gb":6,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":2,"max_seq_length":2048},"lora":{"min_vram_gb":14,"lora_rank":32,"lora_alpha":64,"lr":1e-4,"batch_size":4,"max_seq_length":4096},"full":{"min_vram_gb":32,"lr":5e-5,"batch_size":2,"max_seq_length":4096}},
        "notes":["金融领域专项优化","Baichuan 架构","适合金融客服、风控等场景"],
        "recommended_datasets":[],
        "tags":["金融","中文","垂直领域"],
    },

    # ---- Llama 3.2 系列 ----
    {
        "id": "llama32-3b",
        "patterns": ["llama-3.2-3b", "meta-llama/llama-3.2-3b-instruct"],
        "display_name": "Llama-3.2-3B-Instruct",
        "family": "llama", "chat_template": "llama3",
        "params_b": 3.0, "size_gb": 6.0, "license": "🟢 Llama 3.2",
        "description": "Meta 轻量多语言模型，3B参数，移动端友好",
        "sources": [{"platform":"huggingface","id":"meta-llama/Llama-3.2-3B-Instruct"}],
        "training": {"qlora":{"min_vram_gb":4,"lora_rank":8,"lora_alpha":16,"lr":5e-4,"batch_size":4,"max_seq_length":1024},"lora":{"min_vram_gb":8,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":8,"max_seq_length":2048},"full":{"min_vram_gb":14,"lr":1e-4,"batch_size":4,"max_seq_length":2048}},
        "notes":["支持多语言","Llama3 Chat Template","轻量适合入门"],
        "recommended_datasets":[],
        "tags":["多语言","轻量","入门"],
    },

    # ---- Mixtral 系列 ----
    {
        "id": "mixtral-8x7b",
        "patterns": ["mixtral", "mixtral-8x7b", "mistralai/mixtral-8x7b-instruct"],
        "display_name": "Mixtral-8x7B-Instruct-v0.1",
        "family": "mistral", "chat_template": "mistral",
        "params_b": 46.7, "size_gb": 93.0, "license": "🟢 Apache 2.0",
        "description": "Mistral AI MoE 模型，8专家×7B，英文能力超强",
        "sources": [{"platform":"huggingface","id":"mistralai/Mixtral-8x7B-Instruct-v0.1"}],
        "training": {"qlora":{"min_vram_gb":16,"lora_rank":32,"lora_alpha":64,"lr":1e-4,"batch_size":1,"max_seq_length":1024},"lora":{"min_vram_gb":32,"lora_rank":64,"lora_alpha":128,"lr":5e-5,"batch_size":1,"max_seq_length":2048},"full":{"min_vram_gb":999,"lr":2e-5,"batch_size":1,"max_seq_length":512}},
        "notes":["MoE架构，46.7B参数但每token只激活12.9B","英文顶级","QLoRA在24GB可跑"],
        "recommended_datasets":[],
        "tags":["MoE","英文","高性能"],
    },

    # ---- Gemma 2 系列 (补充 9B/27B) ----
    {
        "id": "gemma-2-9b",
        "patterns": ["gemma-2-9b", "google/gemma-2-9b-it"],
        "display_name": "Gemma-2-9B-Instruct",
        "family": "gemma", "chat_template": "gemma",
        "params_b": 9.0, "size_gb": 18.0, "license": "🟢 Gemma",
        "description": "Google Gemma2 9B，英文表现接近更大模型",
        "sources": [{"platform":"huggingface","id":"google/gemma-2-9b-it"}],
        "training": {"qlora":{"min_vram_gb":6,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":2,"max_seq_length":2048},"lora":{"min_vram_gb":16,"lora_rank":32,"lora_alpha":64,"lr":1e-4,"batch_size":4,"max_seq_length":4096},"full":{"min_vram_gb":42,"lr":5e-5,"batch_size":2,"max_seq_length":4096}},
        "notes":["英文能力非常强","Gemma Chat Template","比 2B 版本强很多"],
        "recommended_datasets":[],
        "tags":["英文","对话","Google"],
    },

    # ---- CodeLlama 系列 ----
    {
        "id": "codellama-7b",
        "patterns": ["codellama", "codellama-7b", "codellama/codellama-7b-instruct"],
        "display_name": "CodeLlama-7B-Instruct",
        "family": "llama", "chat_template": "llama2",
        "params_b": 7.0, "size_gb": 14.0, "license": "🟢 Llama 2",
        "description": "Meta 代码专用模型，编程任务表现优异",
        "sources": [{"platform":"huggingface","id":"codellama/CodeLlama-7b-Instruct-hf"}],
        "training": {"qlora":{"min_vram_gb":6,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":2,"max_seq_length":2048},"lora":{"min_vram_gb":16,"lora_rank":32,"lora_alpha":64,"lr":1e-4,"batch_size":4,"max_seq_length":4096},"full":{"min_vram_gb":40,"lr":5e-5,"batch_size":2,"max_seq_length":4096}},
        "notes":["代码生成和补全专长","支持多种编程语言","FIM填空模式"],
        "recommended_datasets":["code-alpaca"],
        "tags":["代码","编程","FIM"],
    },

    # ---- DeepSeek-R1 系列 ----
    {
        "id": "deepseek-r1-distill-qwen-7b",
        "patterns": ["deepseek-r1", "deepseek-r1-distill-qwen", "deepseek-ai/deepseek-r1-distill-qwen-7b"],
        "display_name": "DeepSeek-R1-Distill-Qwen-7B",
        "family": "qwen2", "chat_template": "chatml",
        "params_b": 7.0, "size_gb": 14.5, "license": "🟢 MIT",
        "description": "DeepSeek-R1 蒸馏版，通过蒸馏获得推理链能力",
        "sources": [{"platform":"huggingface","id":"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}],
        "training": {"qlora":{"min_vram_gb":6,"lora_rank":16,"lora_alpha":32,"lr":2e-4,"batch_size":2,"max_seq_length":2048},"lora":{"min_vram_gb":16,"lora_rank":32,"lora_alpha":64,"lr":1e-4,"batch_size":4,"max_seq_length":4096},"full":{"min_vram_gb":40,"lr":5e-5,"batch_size":2,"max_seq_length":4096}},
        "notes":["具备推理链(Chain-of-Thought)能力","蒸馏自 DeepSeek-R1","Qwen架构，ChatML模板"],
        "recommended_datasets":[],
        "tags":["推理","蒸馏","中文"],
    },

    # ---- Qwen2.5-72B 系列 ----
    {
        "id": "qwen25-72b",
        "patterns": ["qwen2.5-72b", "qwen/qwen2.5-72b-instruct"],
        "display_name": "Qwen2.5-72B-Instruct",
        "family": "qwen2", "chat_template": "chatml",
        "params_b": 72.0, "size_gb": 144.0, "license": "🟢 Apache 2.0",
        "description": "Qwen2.5 旗舰版，72B参数，中文能力顶级",
        "sources": [{"platform":"modelscope","id":"Qwen/Qwen2.5-72B-Instruct"},{"platform":"huggingface","id":"Qwen/Qwen2.5-72B-Instruct"}],
        "training": {"qlora":{"min_vram_gb":16,"lora_rank":32,"lora_alpha":64,"lr":1e-4,"batch_size":1,"max_seq_length":512},"lora":{"min_vram_gb":40,"lora_rank":64,"lora_alpha":128,"lr":5e-5,"batch_size":1,"max_seq_length":1024},"full":{"min_vram_gb":999,"lr":2e-5,"batch_size":1,"max_seq_length":512}},
        "notes":["⛔ 消费级显卡无法全参微调","QLoRA在24GB A5000可跑","云端A100推荐"],
        "recommended_datasets":[],
        "tags":["旗舰","中文","大模型","云端"],
    },
]

def load_cards() -> list[dict]:
    """加载所有适配卡片（优先从 JSON 文件，fallback 到默认库）"""
    if CARDS_FILE.exists():
        try:
            with open(CARDS_FILE, encoding="utf-8") as f:
                cards = json.load(f)
                if cards:
                    return cards
        except Exception:
            pass

    # Save defaults for next time
    save_cards_to_file(DEFAULT_CARDS)
    return DEFAULT_CARDS


def save_cards_to_file(cards: list[dict]):
    """保存卡片库到本地 JSON"""
    CARDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CARDS_FILE, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)


def match_card(model_id: str) -> dict | None:
    """
    根据模型 ID 匹配适配卡片。

    匹配规则：model_id 包含 patterns 中的任一关键词。
    """
    model_id_lower = model_id.lower()
    cards = load_cards()

    # Find ALL matches, prefer the most specific (longest pattern)
    best_match = None
    best_len = 0
    for card in cards:
        for pattern in card["patterns"]:
            if pattern in model_id_lower and len(pattern) > best_len:
                best_match = card
                best_len = len(pattern)

    return best_match


def get_card_by_family(family: str) -> dict | None:
    """根据模型 family 查找适配卡片"""
    cards = load_cards()
    for card in cards:
        if card["family"] == family:
            return card
    return None


def search_cards(query: str = "", tag: str = "", min_params: float = 0, max_params: float = 999) -> list[dict]:
    """
    搜索适配卡片。

    Args:
        query: 名称/族系关键词
        tag: 标签过滤
        min_params: 最小参数量（B）
        max_params: 最大参数量（B）
    """
    cards = load_cards()
    results = []

    query_lower = query.lower()
    for card in cards:
        # Tag filter
        if tag and tag not in card.get("tags", []):
            continue

        # Param filter
        if card["params_b"] < min_params or card["params_b"] > max_params:
            continue

        # Query match
        if query:
            match = False
            for pattern in card["patterns"]:
                if query_lower in pattern:
                    match = True
                    break
            if not match and query_lower in card["display_name"].lower():
                match = True
            if not match and query_lower in card["family"]:
                match = True
            if not match and any(query_lower in t for t in card.get("tags", [])):
                match = True
            if not match:
                continue

        results.append(card)

    # Sort by relevance (params_b ascending)
    results.sort(key=lambda x: x["params_b"])
    return results


def get_training_recommendation(model_id: str, available_vram_gb: float) -> dict:
    """
    给一个模型和可用显存，返回最佳训练方案。

    Returns:
        {
            "card": dict or None,
            "recommended_method": "qlora"|"lora"|"full"|"insufficient",
            "config": {...},  # 推荐参数
            "notes": [...],   # 注意事项
        }
    """
    card = match_card(model_id)

    if not card:
        return {
            "card": None,
            "recommended_method": "qlora",
            "config": {
                "lora_rank": 16, "lora_alpha": 32, "lr": 2e-4,
                "batch_size": 2, "max_seq_length": 2048,
            },
            "notes": ["未找到适配卡片，使用默认参数"],
        }

    training = card["training"]

    # Determine best method based on VRAM
    methods = []
    for method, config in training.items():
        if available_vram_gb >= config["min_vram_gb"]:
            methods.append((method, config))

    if not methods:
        return {
            "card": card,
            "recommended_method": "insufficient",
            "config": None,
            "notes": [
                f"⚠️ 显存不足。此模型至少需要 {training['qlora']['min_vram_gb']}GB 显存（QLoRA）",
                f"当前可用显存：{available_vram_gb:.1f}GB",
                "建议：① 换更小的模型 ② 使用云端 GPU",
            ],
        }

    # Prefer lora > qlora for quality, but qlora > lora if VRAM is tight
    recommended = methods[-1]  # most demanding method that fits
    method_name = recommended[0]
    config = recommended[1].copy()
    config.pop("min_vram_gb", None)

    return {
        "card": card,
        "recommended_method": method_name,
        "config": config,
        "chat_template": card["chat_template"],
        "recommended_datasets": card.get("recommended_datasets", []),
        "notes": card.get("notes", []),
        "tags": card.get("tags", []),
    }


def list_all_families() -> list[str]:
    """列出所有支持的模型族系"""
    cards = load_cards()
    families = set(c["family"] for c in cards)
    return sorted(families)


def get_card_display(card: dict) -> str:
    """生成适配卡片的可读摘要"""
    training = card["training"]
    lines = [
        f"📊 {card['display_name']}",
        f"   参数：{card['params_b']}B · 大小：~{card['size_gb']}GB · {card['license']}",
        f"   架构：{card['family']} · Chat：{card['chat_template']}",
        f"   训练最低要求：",
    ]
    for method, cfg in training.items():
        lines.append(f"     {method.upper():6s} ≥ {cfg['min_vram_gb']:3d}GB 显存")
    if card.get("notes"):
        lines.append(f"   注意事项：")
        for note in card["notes"]:
            lines.append(f"     · {note}")
    return "\n".join(lines)


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    # List all cards
    cards = load_cards()
    print(f"📋 已加载 {len(cards)} 个模型适配卡片\n")

    # Test matching
    test_models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-Coder-6.7B-Instruct",
        "some/unknown-model",
    ]

    for model_id in test_models:
        print(f"🔍 {model_id}")
        rec = get_training_recommendation(model_id, 11.6)
        if rec["card"]:
            print(f"    ✅ 匹配: {rec['card']['display_name']}")
            print(f"    🎯 推荐方式: {rec['recommended_method']}")
            print(f"    ⚙️  参数: rank={rec['config'].get('lora_rank')}, lr={rec['config'].get('lr')}")
        else:
            print(f"    ⚪ 未匹配，使用默认参数")
        print()

    # Search by tag
    print("🏷️  搜索「中文」标签:")
    results = search_cards(tag="中文")
    for r in results:
        print(f"   {r['display_name']} ({r['params_b']}B)")

    print()
    print("✅ 模型适配卡片库自检完成")
