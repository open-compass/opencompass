# 快速开始

本页用同一个模型和同一个数据集展示两种入口：**配置文件**适合保存、复用评测设置，也便于版本管理与结果复现，推荐使用；而 **CLI** 适合快速试跑。此处给出的示例将通过 OpenAI Responses API 评测 `gpt-6-astra` 在 GSM8K 的 64 条演示样本上的表现。

开始前，请先完成[安装与环境准备](installation.md)并进入 OpenCompass 仓库根目录。随后打开 `opencompass/configs/models/openai/gpt_6_astra.py`，选择以下任一种方式配置 API 密钥。

对于临时的本地测试，可以直接将密钥传给模型配置：

```python
from opencompass.models import OpenAISDKResponse

models = [
    dict(
        type=OpenAISDKResponse,
        abbr='gpt-6-astra-response',
        path='gpt-6-astra',
        key='your-api-key',
        # 其余参数不变
    )
]
```

更推荐将密钥保存到自定义环境变量中。例如，先在终端设置 `OPENCOMPASS_API_KEY`：

```bash
export OPENCOMPASS_API_KEY="your-api-key"
```

然后在模型配置中读取该环境变量：

```python
import os
from opencompass.models import OpenAISDKResponse

models = [
    dict(
        type=OpenAISDKResponse,
        abbr='gpt-6-astra-response',
        path='gpt-6-astra',
        key=os.getenv('OPENCOMPASS_API_KEY'),
        # 其余参数不变
    )
]
```

本示例不需要本地 GPU，但运行前应确认当前环境和账号秘钥能够访问相应的 API 服务。首次运行还可能联网下载数据集。

## 路径一：使用配置文件

仓库已经提供 `gpt-6-astra` 模型配置与 GSM8K 演示数据集。新建 `quick_start_eval.py`：

```python
# quick_start_eval.py
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets
    from opencompass.configs.models.openai.gpt_6_astra import \
        models as gpt6_astra_models

datasets = gsm8k_datasets
models = gpt6_astra_models
```

先通过添加 `--dry-run` 参数检查配置能否解析，以及推理任务如何划分。此参数不会真正执行模型推理：

```bash
opencompass quick_start_eval.py --dry-run
```

确认无误后运行完整流程：

```bash
opencompass quick_start_eval.py \
    --work-dir outputs/quick_start_config \
    --debug
```

`--debug` 会在当前进程中依次执行任务并直接打印日志到终端，适合第一次排错；正式批量评测通常不需要它。

## 路径二：直接使用 CLI

不创建配置文件也可以完成同一项评测。从 `opencompass/configs/models` 与 `opencompass/configs/datasets` 中获取配置文件名称（不含 .py 后缀），并作为 `--models` 和 `--datasets` 的参数值传入：

```bash
opencompass \
    --models gpt_6_astra \
    --datasets demo_gsm8k_chat_gen \
    --work-dir outputs/quick_start_cli \
    --debug
```

若要查找配置文件的名称，可运行配置搜索工具：

```bash
python tools/list_configs.py gpt_6_astra gsm8k
```

CLI 方式适合快速验证，但不适合需要精细控制的评测。需要组合多个模型、数据集，修改模型的并发与具体调用参数，以及自定义执行策略时，优先使用配置文件。

## 查看结果

每次运行会在 `--work-dir` 下创建时间戳目录，主要内容包括：

- `configs/`：本次运行最终生效的配置快照；
- `predictions/`：逐样本模型输出；
- `results/`：评测器输出的指标与明细；
- `summary/`：最终汇总结果，包括 CSV、Markdown 等格式。

如果执行失败，先检查终端或时间戳目录中的日志，并参阅“其他文档”章节的[常见问题](../faq/index.md)。完成本页后，建议阅读[工作流与核心概念](../user_guides/framework_overview.md)，再通过[使用配置文件完成一次完整评测](../user_guides/config_based_evaluation.md)学习可复现的完整配置方式。
