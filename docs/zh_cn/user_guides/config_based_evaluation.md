# 使用配置文件完成一次完整评测

本教程从空文件开始，构建一个可解析、可试跑、可复现的评测配置。最终实验通过 OpenAI Responses API 调用 gpt-6-astra，评测 64 条 GSM8K 演示样本，并展示如何显式控制任务划分、执行器和结果汇总。

## 1. 创建配置文件

在仓库根目录新建 `my_eval.py`。先用 `read_base()` 引入仓库已有的模型和数据集配置：

```python
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets
    from opencompass.configs.models.openai.gpt_6_astra import \
        models as gpt6_models

models = gpt6_models
datasets = gsm8k_datasets
```

`models` 和 `datasets` 都必须是列表。通过别名导入模型变量，可以让多个模型配置组合时更易读。`demo_gsm8k_chat_gen` 只选取 64 条测试样本，适合验证环境，不代表正式基准结果。

- 更换本地权重、API 或推理后端：参阅[模型接入](models.md)；
- 选择数据集和配置变体：参阅[数据集配置](datasets.md)；
- 理解 `read_base()`、覆盖和变量拼接：参阅[配置语法与复用参考](config.md)。

## 2. 理解数据集中的输入与评分配置

导入的数据集已经包含 `reader_cfg`、`infer_cfg` 和 `eval_cfg`。它们分别确定读取哪些字段、怎样构造模型输入，以及怎样从输出中计算指标。不要仅凭数据集名称推断评测方法；正式评测时应审阅具体配置文件。

新数据集配置推荐使用 [RawPromptTemplate](../prompt/raw_prompt_template.md) 描述消息。传统的 [PromptTemplate](../prompt/raw_prompt_template.md#prompttemplate传统模板) 仍然有效；模型侧的角色映射由[模型侧对话模板协议](../prompt/meta_template.md)处理。

## 3. 先使用默认策略执行评测任务

上面的最小配置已经能运行：当配置中没有 `infer` 和 `eval` 时，`opencompass` 会补齐本地默认值。先检查最终配置和任务切分：

```bash
opencompass my_eval.py --dry-run --config-verbose
```

`--dry-run` 会解析配置并构造任务，但不会加载模型执行推理。它能尽早发现导入路径、字段名称和资源声明错误。

然后执行完整评测：

```bash
opencompass my_eval.py \
    --work-dir outputs/my_eval \
    --debug
```

调试模式在当前进程中执行并直接显示日志。完成验证后，可去掉 `--debug`，通过 `--max-num-workers` 或者模型配置文件中的参数来调整本地并发。

## 4. 在配置中显式固定任务编排

如果实验需要提交到版本库或长期复现，可以把默认策略明确写入 `my_eval.py`。在文件末尾加入：

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
        task=dict(
            type=OpenICLEvalWatchTask,
            watch_interval=5,
            heartbeat_timeout=60,
        ),
    ),
)
```

这部分描述的是执行方式，不改变数据集中的评分规则：

- Partitioner 决定怎样把模型与数据集拆成任务；
- Runner 决定任务在哪里、以多少并发执行；
- Task 决定执行推理还是读取预测进行评测。

这里将 `num_worker` 和 `max_num_workers` 都设为 1，是因为并发任务由一个进程接管同一模型的全部数据集，再在进程内部调度请求。`OpenICLInferConcurrentTask` 仅适用于 API 模型；本地 GPU 模型应使用 `OpenICLInferTask` 和 `OpenICLEvalTask`。

命令行的 `--slurm` 或 `--dlc` 会用相应运行后端覆盖配置中的执行器。三类组件的职责与并发资源配置详见[任务划分、执行器与任务类型](../execution/tasks_and_runners.md)，API 并发任务的机制见[跨任务并发推理与评测监听](../execution/concurrent_evaluation.md)；多卡与张量并行的模型侧声明见[模型接入总览](models.md)。

## 5. 配置结果汇总

不写 `summarizer` 时会使用默认汇总器。需要固定展示顺序、分组或综合指标时，可以继承已有汇总配置，或显式指定：

```python
from opencompass.summarizers import DefaultSummarizer

summarizer = dict(type=DefaultSummarizer)
```

数据集自身的 Evaluator 负责产生原始指标，Summarizer 负责组织和展示这些指标；二者不要混为一谈。指标含义参阅[评测指标](metrics.md)。

## 6. 运行、恢复与分阶段执行

正式运行：

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

`--mode eval` 和 `--mode viz` 必须配合 `--reuse`，否则 OpenCompass 不知道应读取哪次预测或结果。复用的完整语义与安全边界见[任务恢复、复用与只重跑评测](../execution/reuse_and_resume.md)，全部命令行参数见[命令行参数参考](../execution/cli_reference.md)。

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

验收时至少检查配置快照、失败任务日志、若干条逐样本输入输出和最终指标。只有分数而没有配置、模型版本与数据版本，不能构成可复现的评测记录。

## 8. 扩展为正式实验

在这个配置上继续扩展时，建议每次只改变一类变量：

```python
models = model_group_a + model_group_b
datasets = objective_datasets + subjective_datasets
```

新增或自定义数据集请阅读[新增数据集](../extension/new_dataset.md)与[快速评测自有数据](../extension/custom_dataset.md)；新增模型后端请阅读[新增模型](../extension/new_model.md)。涉及 Judge、数学或代码执行的评测还需要额外依赖和安全边界，不应只替换数据路径后直接运行。
