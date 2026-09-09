# 跨任务并发推理与评测监听

面向大规模 API 模型评测，OpenCompass 提供两个配套组件：**并发推理任务**（`OpenICLInferConcurrentTask`）让一个进程同时推进一个模型的多个数据集；**评测监听任务**（`OpenICLEvalWatchTask`）在推理尚在进行时就开始评分，某个模型—数据集的分片一完成就立即计算指标。二者都只适合大量远端请求的场景，不是本地 GPU 模型的通用加速开关。

## 运行机制总览

两个任务通过**状态文件**和**心跳文件**在同一个工作目录中协作：

```text
主进程 opencompass
  ├─ 心跳线程：每 5 秒向 <work_dir>/infer_heartbeat 写入当前时间
  │
  ├─ 推理：OpenICLInferConcurrentTask（一个进程接管一个模型的全部数据集）
  │     ├─ 把数据集配置中的 Gen/Chat/ChatML Inferencer 替换为 Parallel* 并发版本
  │     ├─ 用全局信号量（max_workers）限制所有数据集共享的请求并发
  │     └─ 持续更新 infer_status/<模型 abbr>/<数据集 abbr>*.json
  │        （status: pending/running/done/fail，附带 total/completed 进度）
  │
  └─ 评分：OpenICLEvalWatchTask（与推理并行启动）
        ├─ 每 watch_interval 秒扫描一遍状态文件
        ├─ 某个模型—数据集的全部分片 status 均为 done → 立刻评分
        └─ 心跳超过 heartbeat_timeout 秒未更新 → 判定推理已停止，
           跳过剩余未完成的数据集并结束
```

## OpenICLInferConcurrentTask

普通 `OpenICLInferTask` 中，一个任务只处理“一个模型 × 一个数据集 × 一个分片”，并行度完全交给 Partitioner 和 Runner。并发任务则相反：**一个进程接管一个模型的所有待测数据集**，在进程内部完成调度。因此配置中 `num_worker=1`、`max_num_workers=1` 是有意为之——数据集不应再被切分或复制成多个任务。

进程内部的工作方式：

1. **API 模型限定**：构建模型后检查 `model.is_api`，本地 GPU 模型直接报错；
2. **数据集调度**：用线程池同时运行多个数据集（上限 `min(数据集数, 32)`），并在“运行中任务的剩余样本总数低于 `2 × max_workers`”时提前启动下一个数据集，避免请求断流；
3. **全局节流**：创建 `max_workers` 个许可的信号量挂到模型上，所有数据集的请求共同受它约束——总压力不随数据集数量增长；
4. **状态上报**：每个“模型 × 数据集”在 `infer_status/` 下维护一个 JSON（多分片时为 `<数据集>_0.json`、`<数据集>_1.json`……），状态流转 pending → running → done/fail，并持续写入进度，这是监听任务的判断依据；
5. **断点续跑**：prediction 文件已存在的模型—数据集组合直接跳过。

`max_workers` 优先取模型配置中的同名字段，未设置时默认 `min(32, CPU 核数 + 4)`。

## 与 Parallel Inferencer 的关系

`ParallelGenInferencer`、`ParallelChatInferencer`、`ParallelChatMLInferencer` 是对应单数据集推理器的并发子类：`batch_size` 固定为 1，用线程池让**同一个数据集内**的多条样本请求同时在飞，逐样本写入 `tmp_*.jsonl` 增量保存（重启后从已完成样本继续），并通过 `progress_tracker` 回报进度。

两层组件的分工：

| 层            | 组件                         | 负责什么                                                 |
| ------------- | ---------------------------- | -------------------------------------------------------- |
| Task 层       | `OpenICLInferConcurrentTask` | 跨数据集：一个进程管多个数据集、共享请求预算、写状态文件 |
| Inferencer 层 | `ParallelGenInferencer` 等   | 数据集内：样本级并发请求、增量落盘、进度回调             |

并发任务在运行时**自动**把数据集配置里的 `GenInferencer`/`ChatInferencer`/`ChatMLInferencer` 替换成对应的 Parallel 版本，并传入 `max_infer_workers` 与进度回调；除这三种之外的推理器（如 `PPLInferencer`、`SCInferencer`）暂时还不支持，会直接报错。

Parallel Inferencer 也可以**单独使用**：在数据集配置的 `infer_cfg` 中直接指定，不引入并发任务——

```python
infer_cfg = dict(
    # ...
    inferencer=dict(type=ParallelGenInferencer, max_infer_workers=16),
)
```

单独使用时只有单数据集内的样本并发，没有跨数据集的共享信号量和状态上报，因此也不能配合评测监听。

## OpenICLEvalWatchTask

监听任务继承自 `OpenICLEvalTask`，评分逻辑（Evaluator、后处理、结果落盘）完全相同，区别只在触发时机：普通评测任务在推理全部结束后统一运行，监听任务则与推理并行启动、随完成随评。

具体流程：

1. 启动时收集 prediction 对应 results 文件尚不存在的模型—数据集组合，先等待 10 秒让心跳文件出现；
2. 每 `watch_interval` 秒（默认 5）扫描一次状态文件：某个组合的**全部分片**状态都是 `done` 才评分；`running`、`fail` 或缺分片的组合继续等待；
3. 心跳判断存活：主进程会在推理期间每 5 秒更新一次 `<work_dir>/infer_heartbeat`。监听侧发现心跳距今超过 `heartbeat_timeout` 秒（默认 60），即认定推理进程已经退出，跳过所有剩余组合并结束——**不会**对不完整的预测文件评分；
4. 等待期间每 `log_interval` 秒（默认 30）打印一次剩余数量。

心跳超时应覆盖“推理任务重启/调度的间隔”；网络文件系统延迟较高时适当调大。推理失败（状态 `fail`）的数据集同样要等到超时才会被跳过，可在 `infer_status/` 中直接查看失败原因。

## 完整配置示例

模型侧声明请求并发，`infer` 与 `eval` 分别使用两个任务：

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

`watch_interval`、`heartbeat_timeout`、`log_interval` 写在 `eval.runner.task` 内部；并发推理任务的 `poll_interval`、`log_interval` 同名参数默认值即可满足多数场景。
