# 贡献指南

欢迎为模型训练室做贡献！

## 如何贡献

### 报告 Bug

1. 在 [Issues](https://github.com/ZBcxy/-model-training-room/issues) 搜索是否已有相同问题
2. 如果没有，创建新 Issue，包含：
   - 环境信息（Python 版本、GPU 型号、OS）
   - 复现步骤
   - 错误日志

### 提交代码

1. Fork 仓库
2. 创建分支：`git checkout -b feature/xxx`
3. 提交前运行测试确保通过：
   ```bash
   python tests/test_e2e.py
   python tests/release_test.py
   ```
4. Push 并创建 Pull Request

### 添加新模型卡片

在 `backend/model_cards.py` 的 `DEFAULT_CARDS` 列表中添加新条目：

```python
{
    "id": "model-id",
    "patterns": ["模型名关键词", "huggingface/model-id"],
    "display_name": "模型显示名",
    "family": "架构族",
    "chat_template": "chatml",
    "params_b": 7.0,
    "size_gb": 14.5,
    "license": "🟢 Apache 2.0",
    "description": "描述",
    "sources": [{"platform":"modelscope","id":"..."}],
    "training": {
        "qlora": {"min_vram_gb":6, "lora_rank":16, ...},
        "lora":  {"min_vram_gb":16, ...},
        "full":  {"min_vram_gb":40, ...},
    },
    "notes": ["注意事项"],
    "recommended_datasets": ["数据集ID"],
    "tags": ["标签"],
}
```

### 添加新训练数据集

在 `backend/dataset_manager.py` 的 `BUILTIN_DATASETS` 中添加。

## 开发环境

```bash
git clone https://github.com/ZBcxy/-model-training-room.git
cd model-training-room
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python auto_setup.py
```
