# Task Partitioning, Runners, and Task Types

OpenCompass divides inference and evaluation into tasks and gives them to a Runner. The three configuration layers are:

- Partitioner: determines how many tasks are created from model-dataset combinations.
- Runner: launches tasks locally or in external environments such as Slurm, Alibaba Cloud, or Volcano Engine with the specified configuration and concurrency.
- Task: performs inference, evaluation, or another concrete operation.

Users control inference and evaluation independently through `infer.partitioner` / `infer.runner` and `eval.partitioner` / `eval.runner`. This page covers Partitioners, Runners, and ordinary task types. For cross-dataset API concurrency internals, see [Cross-Task Concurrent Inference and Evaluation Watching](concurrent_evaluation.md).

## Task Partitioning: Three Partitioners

### NaivePartitioner

Treats every “model × dataset” combination as one independent task. It is the simplest strategy and has no additional arguments. It naturally suits evaluation because the prediction file for each combination is already a whole.

![](https://github.com/user-attachments/assets/f92524ea-5451-429d-a446-97bf36d917ea)

```python
from opencompass.partitioners import NaivePartitioner

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    # ...
)
```

### NumWorkerPartitioner

The default inference partitioner. It divides each dataset into `num_split` shards, then distributes those shards evenly among `num_worker` tasks, so the expected task count matches the actual worker count.

![](https://github.com/user-attachments/assets/432a6738-3298-4729-8b00-a370ea5053ac)
![](https://github.com/user-attachments/assets/07fb30fa-eb2d-4f1b-bf7d-c05ebdba518d)

```python
from opencompass.partitioners import NumWorkerPartitioner

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=16,    # Number of tasks after partitioning / expected workers
        num_split=None,   # Shards per dataset; uses num_worker when None
        min_task_size=16, # Minimum samples in one partition
    ),
    # ...
)
```

`strategy` can be `heuristic` (default) or `split`; the latter only splits large datasets and does not combine small ones.

```{warning}
This partitioner is not suitable for the evaluation stage (`OpenICLEvalTask`).
```

```{warning}
Do not change `num_split` when resuming inference. If `num_split` is `None`, do not change `num_worker`, or existing prediction shards will no longer align for reuse.
```

### SizePartitioner

Estimates each dataset's inference cost by multiplying its size by an expansion coefficient, then splits large datasets and combines small datasets to make subtask costs as even as possible.

![](https://github.com/user-attachments/assets/b707c92f-0738-4e9a-a53e-64510c75898b)

```python
from opencompass.partitioners import SizePartitioner

infer = dict(
    partitioner=dict(
        type=SizePartitioner,
        max_task_size=2000,  # Maximum sample count in one task
        gen_task_coef=20,    # Expansion coefficient for generative tasks
    ),
    # ...
)
```

The cost coefficient depends on inference type: a generative task (`GenInferencer`) uses the larger `gen_task_coef`, while a discriminative task (`PPLInferencer`) uses the number of labels in the prompt. This remains a rough estimate and does not precisely reflect the computational difference between the two task types.

```{warning}
This partitioner is not suitable for the evaluation stage (`OpenICLEvalTask`).
```

### Partitioning and Recovery

Changing partition arguments can make existing prediction files impossible to reuse correctly. Keep model abbreviations, dataset abbreviations, and partitioning strategy stable when resuming. See [Task Recovery, Artifact Reuse, and Evaluation-only Runs](#task-recovery-artifact-reuse-and-evaluation-only-runs).

## Execution Backends: Runner

### LocalRunner

The basic Runner launches tasks in parallel on the local machine:

```python
from opencompass.runners import LocalRunner

runner=dict(
    type=LocalRunner,
    max_num_workers=16,  # Maximum parallel task count; default 16
    max_workers_per_gpu=1,
    task=dict(type=OpenICLInferTask),
)
```

Actual parallel task count is limited by both available GPU resources and `max_num_workers`.

### SlurmRunner

Submits tasks to a Slurm cluster:

```python
from opencompass.runners import SlurmRunner

runner=dict(
    type=SlurmRunner,
    partition='my-partition',  # Cluster partition
    quotatype='auto',          # Optional quota type
    max_num_workers=16,        # Maximum concurrent tasks; default 32
    retry=2,                   # Retry count after task failure
    task=dict(type=OpenICLInferTask),
)
```

### DLCRunner

Submits tasks to Alibaba Cloud PAI-DLC and requires a preconfigured `dlc` CLI and workspace:

```python
from opencompass.runners import DLCRunner

runner=dict(
    type=DLCRunner,
    max_num_workers=16,
    retry=2,
    aliyun_cfg=dict(
        workspace_id='ws-xxx',               # DLC workspace ID
        worker_image='xxx',                  # Task image
        dlc_config_path='/user/.dlc/config', # dlc configuration file
        conda_env_name='opencompass',        # OpenCompass conda environment
    ),
    task=dict(type=OpenICLInferTask),
)
```

### VOLCRunner

Submits tasks to the Volcano Engine Machine Learning Platform. Install and configure the `volc` CLI and prepare a machine-learning task YAML file before use:

```python
from opencompass.runners import VOLCRunner
from opencompass.tasks import OpenICLInferTask

runner=dict(
    type=VOLCRunner,
    queue_name='your-resource-queue',
    max_num_workers=16,
    retry=2,
    preemptible=False,
    volcano_cfg=dict(
        volcano_config_path='/path/to/ml_task.yaml',
        python_env_path='/path/to/opencompass-env',
        hf_offline=True,
        extra_envs=['COMPASS_DATA_CACHE=/path/to/data'],
    ),
    task=dict(type=OpenICLInferTask),
)
```

`VOLCRunner` submits tasks with `volc ml_task submit` and tracks their status and logs through the `get` and `logs` subcommands. It modifies the `Flavor` of the `RoleName: worker` role in the YAML according to the Task's `num_gpus`, so that role must be present. `python_env_path` may instead be replaced by `bashrc_path` and `conda_env_name` to activate a specific Conda environment.

### RJOBRunner

Submits and tracks tasks through the `rjob` CLI for an rjob-scheduled cluster. Resource requirements are derived automatically from task `num_gpus` (GPU count, memory, and CPU); a GPU-free task can specify `memory` and `cpu` directly in `rjob_cfg`. After submission, the Runner polls `rjob get` until completion:

```python
from opencompass.runners import RJOBRunner

runner=dict(
    type=RJOBRunner,
    max_num_workers=16,
    retry=2,
    rjob_cfg=dict(
        task_id='my-exp',               # Used to generate the task name
        image='xxx',                    # Task image
        mount=['/path/to/shared'],      # Mount path, string or list
        env=dict(HF_HOME='/cache/huggingface'),  # Injected environment variables
    ),
    task=dict(type=OpenICLInferTask),
)
```

`rjob_cfg` also supports fields including `charged_group`, `private_machine`, `replicas`, `host_network`, and `extra_args`.

CLI `--slurm` / `--dlc` are runtime overrides: even when a configuration defines a Runner, it is replaced by the requested type and an override warning is printed. `VOLCRunner` has no corresponding CLI override and must be declared explicitly in the configuration file.

## Task Types

A Task is an independent script responsible for compute-intensive operations, with arguments determined by configuration. It can be instantiated and executed with `task.run()`, or produce a complete command through `get_command` for a scheduler, such as `srun {task_cmd}`. Currently supported task types include:

- `OpenICLInferTask`: performs language-model inference through OpenICL.
- `OpenICLEvalTask`: reads predictions and performs evaluation.
- `OpenICLInferConcurrentTask`: lets one process advance multiple datasets concurrently for an API model.
- `OpenICLEvalWatchTask`: watches inference state and evaluates outputs as they complete.

See [Cross-Task Concurrent Inference and Evaluation Watching](concurrent_evaluation.md) for arguments, behavior, and selection guidance for the last two tasks.

## Task Recovery, Artifact Reuse, and Evaluation-only Runs

OpenCompass stores predictions, scores, and summaries separately, so completed stages can be reused. The working directory, timestamp, model and dataset abbreviations, and task partitioning must remain consistent when artifacts are reused.

In the commands below, `--reuse`, `--mode`, and `--work-dir` may be abbreviated as `-r`, `-m`, and `-w`, respectively. Long and short forms may be combined.

### Reuse the Most Recent Run

```bash
opencompass my_eval.py --work-dir outputs/my_eval --reuse
```

Without a value, `--reuse` (or `-r`) selects the last timestamped directory by name under the working directory. To identify the reuse target unambiguously, specify its timestamp:

```bash
opencompass my_eval.py \
    -w outputs/my_eval \
    -r 20260903_120000
```

### Execute Individual Stages

```bash
# Generate predictions only
opencompass my_eval.py -w outputs/my_eval -m infer

# Rescore existing predictions
opencompass my_eval.py -w outputs/my_eval \
    -r 20260903_120000 -m eval

# Regenerate the summary from existing results
opencompass my_eval.py -w outputs/my_eval \
    -r 20260903_120000 -m viz
```

When using `eval` or `viz` mode, specify an existing experiment through `--reuse` or use the result-station read mechanism.
