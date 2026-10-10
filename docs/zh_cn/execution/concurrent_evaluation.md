# 跨任务并发推理与评测监听

本页介绍 OpenCompass 的标准任务类型，重点说明大规模 API 评测的任务编排方式。基础教程默认使用**并发推理任务**（`OpenICLInferConcurrentTask`），由单个进程并发处理同一模型的多个数据集；同时使用**评测监听任务**（`OpenICLEvalWatchTask`），在推理阶段尚未全部完成时对已完成的数据集启动评测。二者配合可减少调度开销，使推理与评测并行执行，适用于请求量较大的 API 评测。

## 运行机制总览

两个任务通过**状态文件**和**心跳文件**在同一个工作目录中协作：

```text
OpenCompass 主进程
  ├─ 心跳线程：每 5 秒向 <work_dir>/infer_heartbeat 写入当前时间
  │
  ├─ 推理：OpenICLInferConcurrentTask（并发处理分配给该模型的数据集）
  │     ├─ 将数据集配置中的 Gen/Chat/ChatML Inferencer 替换为 Parallel* 并发版本
  │     ├─ 使用全局信号量（max_workers）限制所有数据集共享的请求并发
  │     └─ 持续更新 infer_status/<模型 abbr>/<数据集 abbr>*.json
  │        （status: pending/running/done/fail，包含 total/completed 进度）
  │
  └─ 评分：OpenICLEvalWatchTask（与推理并行启动）
        ├─ 每 watch_interval 秒扫描一次状态文件
        ├─ 某个模型—数据集的全部分片 status 均为 done → 启动评测
        └─ 心跳超过 heartbeat_timeout 秒未更新 → 判定推理已停止，
           跳过剩余未完成的数据集并结束
```

## OpenICLInferConcurrentTask

`OpenICLInferConcurrentTask` 在 Task 实例内并发处理 Partitioner 分配给同一模型的数据集。推荐将 `NumWorkerPartitioner` 的 `num_worker` 设为 1，使每个模型的待测数据集集中到一个 Task 中且不再切分；`LocalRunner.max_num_workers` 指定同时运行的 Task 数量上限，单模型评测可设为 1，多模型评测则可根据资源和限流情况适当提高。

内部执行流程如下：

1. **API 模型限定**：模型构建完成后检查 `model.is_api`，仅支持该属性为 `True` 的模型。
2. **数据集调度**：使用线程池并发处理多个数据集（上限为 `min(数据集数, 32)`）；当运行中任务的剩余样本总数低于 `2 × max_workers` 时，提前启动下一个数据集。
3. **状态上报**：每个“模型 × 数据集”在 `infer_status/` 下维护一个 JSON 文件（多分片时为 `<数据集>_0.json`、`<数据集>_1.json`……），持续记录 pending → running → done/fail 的状态变化及执行进度，供监听任务判断是否启动评测。
4. **断点续跑**：若模型—数据集组合的预测文件已存在，则跳过该组合。

`max_workers` 优先读取模型配置中的同名字段；未设置时，默认值为 `min(32, CPU 核数 + 4)`。

并发控制及相关参数如下：

| 参数            | 默认值                  | 配置位置与说明                                    |
| --------------- | ----------------------- | ------------------------------------------------- |
| `max_workers`   | `min(32, CPU 核数 + 4)` | 模型配置字段，控制全部数据集共享的请求并发信号量  |
| `poll_interval` | 1.0                     | Task 配置的顶层字段，指定数据集调度轮询间隔（秒） |
| `log_interval`  | 30.0                    | Task 配置的顶层字段，指定进度日志输出间隔（秒）   |

### 与 Parallel Inferencer 的关系

`ParallelGenInferencer`、`ParallelChatInferencer` 和 `ParallelChatMLInferencer` 分别是相应单数据集推理器的并发子类：`batch_size` 固定为 1，通过线程池并发处理**同一数据集内**的多个样本请求，并将结果逐样本写入 `tmp_*.jsonl` 以实现增量保存。任务重启后可跳过已经完成的样本，并通过 `progress_tracker` 上报进度。

两层组件的分工：

| 层            | 组件                         | 职责                                             |
| ------------- | ---------------------------- | ------------------------------------------------ |
| Task 层       | `OpenICLInferConcurrentTask` | 跨数据集调度、共享请求配额并写入状态文件         |
| Inferencer 层 | `ParallelGenInferencer` 等   | 在单个数据集内执行样本级并发、增量写入并上报进度 |

并发任务在运行时会**自动**将数据集配置中的 `GenInferencer`、`ChatInferencer` 或 `ChatMLInferencer` 替换为对应的 Parallel 版本，并传入 `max_infer_workers` 与进度回调。其他推理器（如 `PPLInferencer`、`SCInferencer`）目前尚不支持，使用时会抛出异常。

Parallel Inferencer 也可以**单独使用**：在数据集配置的 `infer_cfg` 中直接指定，无需使用并发推理任务。

```python
infer_cfg = dict(
    # ...
    inferencer=dict(type=ParallelGenInferencer, max_infer_workers=16),
)
```

单独使用时仅支持单个数据集内的样本级并发，不提供跨数据集共享信号量和状态上报，因此无法配合评测监听任务使用。

## OpenICLEvalWatchTask

监听任务继承自 `OpenICLEvalTask`，使用相同的 Evaluator、后处理和结果保存逻辑，二者仅在启动时机上有所不同：普通评测任务通常在推理全部结束后统一运行，监听任务则与推理任务并行启动，并对已完成的数据集依次进行评测。

执行流程如下：

1. 启动时收集 `results/` 目录下尚无结果文件的模型—数据集组合，并等待 10 秒以确保心跳文件已经创建；
2. 每隔 `watch_interval` 秒（默认 5）扫描一次状态文件：仅当某个组合的**全部分片**状态均为 `done` 时才启动评测；存在 `running`、`fail` 或状态文件缺失的组合将继续等待；
3. **存活状态判断**：主进程在推理期间每 5 秒更新一次 `<work_dir>/infer_heartbeat`。若心跳超过 `heartbeat_timeout` 秒（默认 60）未更新，监听任务将推理进程视为已经退出，跳过所有剩余组合并结束，且**不会**对不完整的预测文件进行评分；
4. 等待期间每隔 `log_interval` 秒（默认 30）输出一次待评测组合的数量。

`heartbeat_timeout` 应大于推理任务重启或重新调度所需的时间；使用延迟较高的网络文件系统时，可适当增大该值。推理失败（状态为 `fail`）的数据集同样会在心跳超时后被跳过，可在 `infer_status/` 中查看失败原因。

以下构造参数配置于 `eval.runner.task`：

| 参数                | 默认值 | 说明                                     |
| ------------------- | ------ | ---------------------------------------- |
| `watch_interval`    | 5.0    | 状态文件扫描间隔（秒）                   |
| `heartbeat_timeout` | 60.0   | 心跳超时（秒），超时后跳过剩余数据集     |
| `log_interval`      | 30.0   | 等待期间输出剩余任务数量的日志间隔（秒） |

## 推理与评测任务配置示例

在模型配置中声明请求并发数，并在 `infer` 与 `eval` 中分别指定推理任务和评测任务：

```python
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalWatchTask, OpenICLInferConcurrentTask

models = [
    dict(
        # ... API 模型的其余字段 ...
        max_workers=16,  # 单任务内所有数据集共享的请求并发
    )
]

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
        max_num_workers=4,
        task=dict(
            type=OpenICLEvalWatchTask,
            watch_interval=5,       # 状态文件扫描间隔（秒）
            heartbeat_timeout=60,   # 心跳超时（秒），超过则跳过剩余数据集
            log_interval=30,        # 等待日志间隔（秒）
        ),
    ),
)
```

`watch_interval`、`heartbeat_timeout`、`log_interval` 配置于 `eval.runner.task`；并发推理任务的 `poll_interval` 和 `log_interval` 使用默认值即可满足多数场景。

## 普通推理与评测任务

`OpenICLInferTask` 会依次处理 Partitioner 分配给同一 Task 的模型与数据集组合，适用于本地一站式部署与评测的任务。任务的组合与切分方式由 Partitioner 决定，Runner 则控制这些 Task 的执行环境和并发数量。

`OpenICLEvalTask` 读取已有预测结果并计算评测指标。它与 `OpenICLEvalWatchTask` 使用相同的评分逻辑，区别在于普通任务通常在推理结束后统一启动，监听任务则根据推理进度对已完成的数据集启动评测。

一般选择如下：

- 大规模 API 评测：`OpenICLInferConcurrentTask` + `OpenICLEvalWatchTask`；
- 本地一站式任务：`OpenICLInferTask` + `OpenICLEvalTask`。
