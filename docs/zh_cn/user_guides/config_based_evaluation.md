# 使用配置文件完成一次完整评测

本教程从空文件开始构建一个完整评测配置。最终实验通过 OpenAI Responses API 调用 gpt-6-astra，评测 64 条 GSM8K 演示样本，并展示如何显式控制任务划分、执行策略，并汇总最终评测结果。

## 1. 创建配置文件

在仓库根目录新建 `my_eval.py`。先用 `read_base()` 引入仓库已有的模型和数据集配置：

```python
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets
    from opencompass.configs.models.openai.gpt_6_astra import \
        models as gpt6_astra_models

datasets = gsm8k_datasets
models = gpt6_astra_models
```

`models` 和 `datasets` 都必须是列表。通过别名导入模型变量，可以让多个模型配置组合时更易读。`demo_gsm8k_chat_gen` 只选取 64 条测试样本，适合验证环境，不代表正式基准结果。

- 更换本地模型权重、API 或推理后端：参阅[模型接入](models.md)；
- 更换数据集、检查数据集的详细配置情况：参阅[数据集配置](datasets.md)；
- 理解 `read_base()`、覆盖和变量拼接：参阅[配置语法与复用参考](config.md)。

## 2. 使用默认策略执行评测任务

上文提供的示例已经能够作为 `opencompass` 的最小合法配置运行。当配置中没有显式指定 `infer` 和 `eval` 字段时，`opencompass` 会补齐本地默认值。使用如下指令来检查最终配置和任务切分策略：

```bash
opencompass my_eval.py --dry-run --config-verbose
```

然后执行完整评测：

```bash
opencompass my_eval.py \
    --work-dir outputs/my_eval \
    --debug
```

## 3. 配置执行策略

你可以通过配置 `infer` 和 `eval` 字段，显式指定评测任务的切分策略、运行环境和任务类型。示例如下：

```python
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalWatchTask, OpenICLInferConcurrentTask

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLInferConcurrentTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLEvalWatchTask),
    ),
)
```

这部分描述的是执行方式，不改变数据集中的评分规则：

- Partitioner 决定按何种策略把模型与数据集组合划分为子任务。`NumWorkerPartitioner` 将待测数据集尽量均匀地分配给不超过 `num_worker` 个子任务；设为 `num_worker=1` 时，同一模型的所有待测数据集会交给一个子任务处理。
- Runner 决定任务在何种环境中运行。`LocalRunner` 在本机启动任务，`max_num_workers=1` 表示该 Runner 同一时间最多运行一个任务。该参数控制任务级并发，不控制单个任务内部的 API 请求并发。
- Task 是推理或评测阶段的实际执行单元，定义该阶段的任务入口与处理逻辑。`OpenICLInferConcurrentTask` 是执行推理的任务实例，通过模型侧的 `max_workers` 控制并发数，但仅适用于 API 模型；直接在评测进程中加载的本地模型应使用 `OpenICLInferTask`。`OpenICLEvalWatchTask` 是执行评测的任务示例，通过监听与心跳机制实现数据集推理完成后的实时评测；本地模型通常与 `OpenICLEvalTask` 搭配使用。

三类组件的职责、主要参数和选择方法详见[任务划分、执行器与任务类型](../execution/tasks_and_runners.md)，以及[跨任务并发推理与评测监听](../execution/concurrent_evaluation.md)。

## 4. 配置结果汇总

通过 `summarizer` 可以指定最终汇总表中的展示内容、指标和顺序。示例如下：

```python
summarizer = dict(
    dataset_abbrs=[
        'All Results',
        ['demo_gsm8k', 'accuracy'],
    ],
)
```

`dataset_abbrs` 中的各项会按列表顺序显示：支持显示类似 `All Results` 的纯文本行，`['demo_gsm8k', 'accuracy']` 指定要展示的数据集 `abbr` 及其指标。Summarizer 只负责组织和展示 Evaluator 产出的指标，不会改变评分规则。

## 5. 运行、恢复与分阶段执行

正式运行命令如下（也可将 `work_dir` 写入配置文件以便复用）：

```bash
opencompass my_eval.py --work-dir outputs/my_eval
```

每次启动会在 `outputs/my_eval/` 下创建时间戳目录。如果某次运行中断，可复用最近一次实验中的已有产物：

```bash
opencompass my_eval.py --work-dir outputs/my_eval --reuse
```

也可以指定时间戳，并只执行某个阶段：

```bash
opencompass my_eval.py \
    --work-dir outputs/my_eval \
    --reuse 20260903_120000 \
    --mode eval
```

`--mode eval` 和 `--mode viz` 必须配合 `--reuse`，否则 OpenCompass 不知道应读取哪次预测或结果。复用逻辑详见[任务恢复、复用与只重跑评测](../execution/tasks_and_runners.md#任务恢复复用与只重跑评测)，全部命令行参数见[命令行参数参考](../execution/cli_reference.md)。

## 6. 完整配置文件的结构

将前面各部分合并后，完整的 `my_eval.py` 如下：

```python
from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import (OpenICLEvalWatchTask,
                               OpenICLInferConcurrentTask)

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets
    from opencompass.configs.models.openai.gpt_6_astra import \
        models as gpt6_astra_models

datasets = gsm8k_datasets
models = gpt6_astra_models

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLInferConcurrentTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLEvalWatchTask),
    ),
)

summarizer = dict(
    dataset_abbrs=[
        'All Results',
        ['demo_gsm8k', 'accuracy'],
    ],
)

work_dir = 'outputs/my_eval'
```

其中，命令行传入的 `--work-dir` 会覆盖配置文件中的 `work_dir`。保存文件后，可以先运行 `opencompass my_eval.py --dry-run --config-verbose` 检查最终配置和任务划分，再启动完整评测。

## 7. 检查产物

一次运行的时间戳目录包含（各目录产物的结构细节见[理解输出与结果汇总](results_and_summarizer.md)）：

```text
outputs/my_eval/<时间戳>/
├── configs/       # 实际生效的配置快照
├── logs/          # 非 debug 模式下的任务日志
├── predictions/   # 逐样本模型输出
├── results/       # 数据集评测结果与明细
└── summary/       # 汇总表格和 CSV 等文件
```

## 8. 扩展评测配置

你可以在配置文件中写入多模型、多评测集、统一配置 Judge 模型、批量化设置上下文等多项扩展方案。例如：

```python
# 合并前面通过 read_base() 导入的模型和数据集配置
models = model_group_a + model_group_b
datasets = objective_datasets + subjective_datasets

# Judge 模型仅用于评分，不属于上面的被测模型列表
from opencompass.models import OpenAISDK
judge_cfg = dict(
    abbr='judge-model',
    type=OpenAISDK,
    path='your-judge-model',
    key='your-judge-key',
    openai_api_base='your-judge-api',
    batch_size=64,
    temperature=0.001,
    max_out_len=16384,
    max_seq_len=262144,
)

for dataset in datasets:
    # 统一覆盖数据集推理器的最大输出长度
    inferencer = dataset.get('infer_cfg', {}).get('inferencer')
    if inferencer is not None:
        inferencer['max_out_len'] = 128000

    # 覆盖直接使用 Judge 模型的 Evaluator 配置
    evaluator = dataset.get('eval_cfg', {}).get('evaluator', {})
    if 'judge_cfg' in evaluator:
        evaluator['judge_cfg'] = judge_cfg
    # 兼容在级联评测器中嵌套的 LLM Evaluator
    if ('llm_evaluator' in evaluator
            and 'judge_cfg' in evaluator['llm_evaluator']):
        evaluator['llm_evaluator']['judge_cfg'] = judge_cfg
```
