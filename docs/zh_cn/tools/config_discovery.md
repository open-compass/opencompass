# 配置发现与比较

## 列出和搜索配置

```bash
python tools/list_configs.py
python tools/list_configs.py qwen mmlu
```

脚本搜索模型、数据集及汇总配置，并输出可传给 `opencompass --models/--datasets/--summarizer` 的简称。搜索结果仍需人工审阅具体文件，尤其是 Prompt、模型版本和评测器。

## 比较配置目录

```bash
python tools/compare_configs.py folder_a folder_b \
    --extensions .py .json .md \
    --ignore folder_a/generated
```

该工具按相对路径比较两个目录：第一个目录独有文件或同名内容不同会导致失败，第二个目录独有文件会打印提示。它适合检查配置集同步，不会理解 Python 配置继承后的语义差异。

评审单个实验的最终生效配置时，应查看输出目录的 `configs/` 快照并配合 `--config-verbose`。
