# 任务划分、执行器与任务类型

OpenCompass 把推理和评测拆成任务后交给 Runner 执行。配置中的三个层次分别是：

- Partitioner：把模型和数据集组合拆成多少个任务；
- Runner：在本地、Slurm 或 DLC 上以多少并发启动任务；
- Task：执行推理、评测或其他具体工作。

大模型推理耗时长、数据集量大，串行运行一次评测的开销往往很大。Partitioner 将大任务按策略切成众多独立的小任务并行运行，是加速评测的主要手段；Runner 决定这些任务在哪里执行。用户通过配置文件中的 `infer.partitioner` / `infer.runner` 和 `eval.partitioner` / `eval.runner` 分别控制推理和评测两个阶段。

## 默认策略

未显式写 `infer`/`eval` 时，CLI 会生成本地默认配置：推理使用 `NumWorkerPartitioner` 和 `OpenICLInferTask`，评测使用 `NaivePartitioner` 和 `OpenICLEvalTask`，两阶段均由 `LocalRunner` 执行。

```python
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=LocalRunner,
        max_num_workers=4,
        max_workers_per_gpu=1,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=4,
        task=dict(type=OpenICLEvalTask),
    ),
)
```

## 任务划分：三种 Partitioner

### NaivePartitioner

将每个“模型 × 数据集”组合作为一个独立任务派发，是最基础的划分策略，没有额外参数。评测阶段天然适合这种划分——每个组合的预测文件本就是一个整体。

![](https://github.com/user-attachments/assets/f92524ea-5451-429d-a446-97bf36d917ea)

```python
from opencompass.partitioners import NaivePartitioner

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    # ...
)
```

### NumWorkerPartitioner

推理阶段的默认划分器。它把每个数据集划分成 `num_split` 份，再将这些分片均匀分入 `num_worker` 个任务，任务数预期与实际运行的 worker 数一致。

![](https://github.com/user-attachments/assets/432a6738-3298-4729-8b00-a370ea5053ac)
![](https://github.com/user-attachments/assets/07fb30fa-eb2d-4f1b-bf7d-c05ebdba518d)

```python
from opencompass.partitioners import NumWorkerPartitioner

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=16,    # 划分完成后的任务数 / 预期 worker 数
        num_split=None,   # 每个数据集被划分成多少份；None 时取 num_worker
        min_task_size=16, # 每个划分的最小样本数
    ),
    # ...
)
```

`strategy` 参数可选 `heuristic`（默认）与 `split`，后者只切分大数据集、不合并小数据集。

```{warning}
该划分器不适用于评测阶段（`OpenICLEvalTask`）。
```

```{warning}
推理需要断点继续时，不要修改 `num_split` 的值；若 `num_split` 为 `None`，则不要修改 `num_worker`，否则已有预测分片无法对齐复用。
```

### SizePartitioner

根据数据集大小乘上扩张系数，估算每个数据集的推理成本，然后通过切分大数据集、合并小数据集的方式创建任务，尽量让各子任务的成本均等。

![](https://github.com/user-attachments/assets/b707c92f-0738-4e9a-a53e-64510c75898b)

```python
from opencompass.partitioners import SizePartitioner

infer = dict(
    partitioner=dict(
        type=SizePartitioner,
        max_task_size=2000,  # 单个任务的最大样本数
        gen_task_coef=20,    # 生成式任务的扩张系数
    ),
    # ...
)
```

估算成本时按推理类型选择系数：生成式任务（`GenInferencer`）使用较大的 `gen_task_coef`；判别式任务（`PPLInferencer`）则使用 prompt 中 label 的数量。这一估算仍然比较粗糙，未能准确反映两类任务的计算量差距。

```{warning}
该划分器不适用于评测阶段（`OpenICLEvalTask`）。
```

### 切分与断点恢复

改变切分参数可能让已有预测文件无法正确复用。需要断点恢复时，应保持模型简称、数据集简称和切分策略稳定，详见[任务恢复、复用与只重跑评测](reuse_and_resume.md)。

## 运行后端：Runner

### LocalRunner

最基本的运行器，将任务在本机并行启动：

```python
from opencompass.runners import LocalRunner

runner=dict(
    type=LocalRunner,
    max_num_workers=16,  # 最大并行任务数，默认 16
    max_workers_per_gpu=1,
    task=dict(type=OpenICLInferTask),
)
```

实际并行任务数同时受可用 GPU 资源和 `max_num_workers` 限制。

### SlurmRunner

将任务提交到 Slurm 集群运行：

```python
from opencompass.runners import SlurmRunner

runner=dict(
    type=SlurmRunner,
    partition='my-partition',  # 集群分区
    quotatype='auto',          # 可选：配额类型
    max_num_workers=16,        # 最大同时运行任务数，默认 32
    retry=2,                   # 任务失败重试次数
    task=dict(type=OpenICLInferTask),
)
```

### DLCRunner

将任务提交到阿里云 PAI-DLC 运行，依赖环境中预先配置好的 dlc 命令行工具和工作空间：

```python
from opencompass.runners import DLCRunner

runner=dict(
    type=DLCRunner,
    max_num_workers=16,
    retry=2,
    aliyun_cfg=dict(
        workspace_id='ws-xxx',               # DLC 工作空间 ID
        worker_image='xxx',                  # 运行任务的镜像
        dlc_config_path='/user/.dlc/config', # dlc 配置文件
        conda_env_name='opencompass',        # OpenCompass 的 conda 环境
    ),
    task=dict(type=OpenICLInferTask),
)
```

### RJOBRunner

通过 rjob 命令行工具提交并跟踪任务，适用于使用 rjob 调度的集群。资源需求根据任务声明的 `num_gpus` 自动推导（GPU 数、内存和 CPU）；无 GPU 的任务可在 `rjob_cfg` 中直接指定 `memory` 和 `cpu`。提交后 Runner 会轮询 `rjob get` 直到任务结束：

```python
from opencompass.runners import RJOBRunner

runner=dict(
    type=RJOBRunner,
    max_num_workers=16,
    retry=2,
    rjob_cfg=dict(
        task_id='my-exp',               # 用于生成任务名
        image='xxx',                    # 运行任务的镜像
        mount=['/path/to/shared'],      # 挂载路径，字符串或列表
        env=dict(HF_HOME='/cache/huggingface'),  # 注入的环境变量
    ),
    task=dict(type=OpenICLInferTask),
)
```

`rjob_cfg` 还支持 `charged_group`、`private_machine`、`replicas`、`host_network` 和 `extra_args` 等字段。

命令行的 `--slurm` / `--dlc` 属于运行时覆盖：即使配置已定义执行器，也会把 Runner 替换成对应类型，并给出覆盖警告。

## 任务类型（Task）

任务是一个独立脚本，负责计算密集的操作，通过配置文件确定参数。它可以实例化后调用 `task.run()` 执行，也可以通过 `get_command` 生成完整命令（如 `srun {task_cmd}`）交给调度系统。目前支持：

- `OpenICLInferTask`：基于 OpenICL 框架执行语言模型推理；
- `OpenICLEvalTask`：读取预测结果执行评测计算。

## 资源声明与实际并发

模型的 `run_cfg.num_gpus` 声明一个任务占用的 GPU 数量；Runner 的 `max_num_workers` 限制同时运行的任务数；LocalRunner 的 `max_workers_per_gpu` 允许一张卡承载几个任务。这三个参数共同决定并发，不能互相替代。

先用 `--dry-run` 检查切分结果。并发增大前还应确认显存、CPU、文件描述符、API 限流和数据缓存能够承受对应负载。
