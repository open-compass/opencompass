# 五分钟快速开始

本页用同一个模型和同一个数据集展示两种入口：**配置文件**适合保存、复用评测设置，也便于版本管理与结果复现，推荐使用；而 **CLI** 适合快速试跑。此处给出的示例将评测 `Qwen/Qwen3.5-35B-A3B`（通过 vLLM 以张量并行加载）在 GSM8K 的 64 条演示样本上的表现。

开始前请先完成[安装与环境准备](installation.md)，进入 OpenCompass 仓库根目录，并准备能够加载该模型的 GPU（配置使用 `tensor_parallel_size=2`）。首次运行可能需要联网下载模型和数据集。

请注意，如果你的环境中没有 GPU，或只希望运行 API 评测，则可以改用例如 `opencompass/configs/models/qwen3/qwen3_8_max.py` 的 API 模型配置。

## 路径一：使用配置文件

仓库已经提供可运行的配置 `examples/eval_chat_demo.py`。它通过 `read_base()` 组合两个模型和两个演示数据集。为了让第一次运行更快，可以复制一份配置，只保留 Qwen3.5 和 GSM8K：

```python
# quick_start_eval.py
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets
    from opencompass.configs.models.qwen3.vllm_qwen3_5_35b_a3b import \
        models

datasets = gsm8k_datasets
```

先检查配置能否解析、任务如何划分。`--dry-run` 不会真正执行模型推理：

```bash
opencompass quick_start_eval.py --dry-run
```

确认无误后运行完整流程：

```bash
opencompass quick_start_eval.py \
    --work-dir outputs/quick_start_config \
    --debug
```

`--debug` 会在当前进程中依次执行任务并直接显示日志，适合第一次排错；正式批量评测通常不需要它。

## 路径二：直接使用 CLI

不创建配置文件也可以完成同一项评测。模型和数据集名称来自 `opencompass/configs/models` 与 `opencompass/configs/datasets`：

```bash
opencompass \
    --models vllm_qwen3_5_35b_a3b \
    --datasets demo_gsm8k_chat_gen \
    --work-dir outputs/quick_start_cli \
    --debug
```

同样可以在执行前加 `--dry-run`。若要查找配置名称，运行：

```bash
python tools/list_configs.py qwen3_5 gsm8k
```

CLI 方式适合快速验证，但不适合需要精细控制的评测。需要组合多个模型、数据集，修改模型的并发与具体调用参数，以及自定义执行策略时，优先使用配置文件。

## 查看结果

每次运行会在 `--work-dir` 下创建时间戳目录，主要内容包括：

- `configs/`：本次运行最终生效的配置快照；
- `predictions/`：逐样本模型输出；
- `results/`：评测器计算出的指标与明细；
- `summary/`：终端表格以及 CSV 等汇总结果。

如果执行失败，先检查时间戳目录中的日志，并参阅[常见问题与故障排查](../faq/index.md)。完成本页后，建议阅读[工作流与核心概念](../user_guides/framework_overview.md)，再通过[使用配置文件完成一次完整评测](../user_guides/config_based_evaluation.md)学习可复现的正式配置方式。
