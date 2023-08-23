# Efficient Evaluation

OpenCompass supports custom task partitioners (`Partitioner`), which enable flexible division of evaluation tasks. In conjunction with `Runner`, which controls the platform for task execution, such as a local machine or a cluster, OpenCompass can distribute large evaluation tasks to a vast number of computing nodes. This helps utilize computational resources efficiently and significantly accelerates the evaluation process.

By default, OpenCompass hides these details from users and automatically selects the recommended execution strategies. But users can still customize these strategies of the workflows according to their needs, just by adding the `infer` and/or `eval` fields to the configuration file:

```python
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=5000),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask)),
)
```

The example above demonstrates the way to configure the execution strategies for the inference and evaluation stages. At the inference stage, the task will be divided into several sub-tasks, each of 5000 samples, and then submitted to the Slurm cluster for execution, where there are at most 64 tasks running in parallel. At the evaluation stage, each single model-dataset pair forms a task, and 32 processes are launched locally to compute the metrics.

The following sections will introduce the involved modules in detail.

## Task Partition (Partitioner)

Due to the long inference time of large language models and the vast amount of evaluation datasets, serial execution of a single evaluation task can be quite time-consuming. OpenCompass allows custom task partitioners (`Partitioner`) to divide large evaluation tasks into numerous independent smaller tasks, thus fully utilizing computational resources via parallel execution. Users can configure the task partitioning strategies for the inference and evaluation stages via `infer.partitioner` and `eval.partitioner`. Below, we will introduce all the partitioning strategies supported by OpenCompass.

### `NaivePartitioner`

This partitioner dispatches each combination of a model and dataset as an independent task. It is the most basic partitioning strategy and does not have any additional parameters.

```python
from opencompass.partitioners import NaivePartitioner

infer = dict(
    partitioner=dict(type=NaivePartitioner)
    # ...
)
```

### `SizePartitioner`

```{warning}
For now, this partitioner is not suitable for evaluation tasks (`OpenICLEvalTask`).
```

This partitioner estimates the inference cost (time) of a dataset according to its size, multiplied by an empirical expansion coefficient. It then creates tasks by splitting larger datasets and merging smaller ones to ensure the inference costs of each sub-task are as equal as possible.

The commonly used parameters for this partitioner are as follows:

```python
from opencompass.partitioners import SizePartitioner

infer = dict(
    partitioner=dict(
        type=SizePartitioner,
        max_task_size: int = 2000,  # Maximum size of each task
        gen_task_coef: int = 20,  # Expansion coefficient for generative tasks
    ),
    # ...
)
```

`SizePartitioner` estimates the inference cost of a dataset based on the type of the inference task and selects different expansion coefficients accordingly. For generative tasks, such as those using `GenInferencer`, a larger `gen_task_coef` is set; for discriminative tasks, like those using `PPLInferencer`, the number of labels in the prompt is used.

```{note}
Currently, this partitioning strategy is still rather crude and does not accurately reflect the computational difference between generative and discriminative tasks. We look forward to the community proposing better partitioning strategies :)
```

## Execution Backend (Runner)

In a multi-card, multi-machine cluster environment, if we want to implement parallel execution of multiple tasks, we usually need to rely on a cluster management system (like Slurm) for task allocation and scheduling. In OpenCompass, task allocation and execution are uniformly handled by the Runner. Currently, it supports both Slurm and PAI-DLC scheduling backends, and also provides a `LocalRunner` to directly launch tasks on the local machine.

### `LocalRunner`

`LocalRunner` is the most basic runner that can run tasks parallelly on the local machine.

```python
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

infer = dict(
    # ...
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,  # Maximum number of processes to run in parallel
        task=dict(type=OpenICLEvalTask),  # Task to be run
    )
)
```

```{note}
The actual number of running tasks are both limited by the actual available GPU resources and the number of workers.
```

### `SlurmRunner`

`SlurmRunner` submits tasks to run on the Slurm cluster. The commonly used configuration fields are as follows:

```python
from opencompass.runners import SlurmRunner
from opencompass.tasks import OpenICLInferTask

infer = dict(
    # ...
    runner=dict(
        type=SlurmRunner,
        task=dict(type=OpenICLEvalTask),  # Task to be run
        max_num_workers=16,  # Maximum concurrent evaluation task count
        retry=2,  # Retry count for failed tasks, can prevent accidental errors
    ),
)
```

### `DLCRunner`

`DLCRunner` submits tasks to run on Alibaba's Deep Learning Center (DLC). This Runner depends on `dlc`. Firstly, you need to prepare `dlc` in the environment:

```bash
cd ~
wget https://dlc-cli.oss-cn-zhangjiakou.aliyuncs.com/light/binary/linux/amd64/dlc
chmod +x ./dlc
sudo ln -rs dlc /usr/local/bin
./dlc config
```

Fill in the necessary information according to the prompts and get the `dlc` configuration file (like `/user/.dlc/config`) to complete the preparation. Then, specify the `DLCRunner` configuration in the configuration file as per the format:

```python
from opencompass.runners import DLCRunner
from opencompass.tasks import OpenICLInferTask

infer = dict(
    # ...
    runner=dict(
        type=DLCRunner,
        task=dict(type=OpenICLEvalTask),  # Task to be run
        max_num_workers=16,  # Maximum concurrent evaluation task count
        aliyun_cfg=dict(
            bashrc_path="/user/.bashrc",  # Path to the bashrc for initializing the running environment
            conda_env_name='opencompass',  # Conda environment for OpenCompass
            dlc_config_path="/user/.dlc/config",  # Configuration file for dlc
            workspace_id='ws-xxx',  # DLC workspace ID
            worker_image='xxx',  # Image url for running tasks
        ),
        retry=2,  # Retry count for failed tasks, can prevent accidental errors
    ),
)
```

## Task

A Task is a fundamental module in OpenCompass, a standalone script that executes the computationally intensive operations. Each task is designed to load a configuration file to determine parameter settings, and it can be executed in two distinct ways:

2. Instantiate a Task object, then call `task.run()`.
3. Call `get_command` method by passing in the config path and the command template string that contains `{task_cmd}` as a placeholder (e.g. `srun {task_cmd}`). The returned command string will be the full command and can be executed directly.

As of now, OpenCompass supports the following task types:

- `OpenICLInferTask`: Perform LM Inference task based on OpenICL framework.
- `OpenICLEvalTask`: Perform LM Evaluation task based on OpenEval framework.

In the future, more task types will be supported.
