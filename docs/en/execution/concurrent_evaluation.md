# Cross-Task Concurrent Inference and Evaluation Watching

This page is a reference for standard OpenCompass task types, with an emphasis on orchestrating large-scale API evaluation. The basic tutorial uses a **concurrent inference task** (`OpenICLInferConcurrentTask`) so one process advances multiple datasets for one model, together with an **evaluation watching task** (`OpenICLEvalWatchTask`) that starts scoring while inference is still running. Both target large numbers of remote requests and are not general acceleration switches for local GPU models.

## Mechanism Overview

The two tasks cooperate through **status files** and a **heartbeat file** in the same work directory:

```text
main opencompass process
  ├─ heartbeat thread: writes the current time to <work_dir>/infer_heartbeat every 5 seconds
  │
  ├─ inference: OpenICLInferConcurrentTask (one process owns every dataset for one model)
  │     ├─ replaces Gen/Chat/ChatML Inferencers with concurrent Parallel* versions
  │     ├─ limits request concurrency shared by all datasets with a global semaphore (max_workers)
  │     └─ continuously updates infer_status/<model abbr>/<dataset abbr>*.json
  │        (status: pending/running/done/fail, with total/completed progress)
  │
  └─ scoring: OpenICLEvalWatchTask (starts in parallel with inference)
        ├─ scans status files every watch_interval seconds
        ├─ all shards for one model-dataset have status done → score immediately
        └─ heartbeat is older than heartbeat_timeout → inference has stopped;
           skip remaining incomplete datasets and exit
```

## Ordinary Inference and Evaluation Tasks

`OpenICLInferTask` treats one “model × dataset × shard” as one task and delegates parallelism to the Partitioner and Runner. It suits local GPU models and partitioned execution on Slurm, DLC, and similar clusters.

`OpenICLEvalTask` reads predictions and computes scores after inference. It uses the same scoring logic as `OpenICLEvalWatchTask`; the ordinary task starts scoring as a group, whereas the Watch task scores shards as they complete and uses a heartbeat timeout to avoid incomplete predictions.

General guidance:

- Large-scale API evaluation: `OpenICLInferConcurrentTask` + `OpenICLEvalWatchTask`.
- Local models or cluster partitioning: `OpenICLInferTask` + `OpenICLEvalTask`.

See [Task Partitioning, Runners, and Task Types](tasks_and_runners.md) for Partitioner and Runner configuration.

## OpenICLInferConcurrentTask

With ordinary `OpenICLInferTask`, one task handles only one “model × dataset × shard,” and parallelism belongs entirely to the Partitioner and Runner. The concurrent task does the opposite: **one process owns every dataset under evaluation for one model** and schedules them internally. Therefore, `num_worker=1` and `max_num_workers=1` are intentional—datasets should not be split or copied into multiple tasks again.

Within the process:

1. **API-model restriction:** after constructing the model, it checks `model.is_api`; a local GPU model raises an error.
2. **Dataset scheduling:** a thread pool runs multiple datasets at once, up to `min(dataset count, 32)`, and starts the next dataset early when the remaining samples in running tasks fall below `2 × max_workers`, preventing request starvation.
3. **Global throttling:** a semaphore with `max_workers` permits is attached to the model. Requests from all datasets share it, so total pressure does not grow with dataset count.
4. **Status reporting:** each “model × dataset” maintains a JSON file under `infer_status/` (`<dataset>_0.json`, `<dataset>_1.json`, and so on for multiple shards). Status moves through pending → running → done/fail and progress is continuously written for the watching task.
5. **Resume:** a model-dataset combination whose prediction file already exists is skipped.

`max_workers` first uses the same-named field from the model configuration. If absent, it defaults to `min(32, CPU core count + 4)`.

Constructor arguments belong under `infer.runner.task`:

| Argument                 | Default                       | Description                                                                            |
| ------------------------ | ----------------------------- | -------------------------------------------------------------------------------------- |
| `poll_interval`          | 1.0                           | Dataset-scheduling poll interval in seconds                                            |
| `log_interval`           | 30.0                          | Progress log interval in seconds                                                       |
| `max_workers`            | `min(32, CPU core count + 4)` | Request-concurrency semaphore shared by all datasets; the model field takes precedence |
| `dump_res_length`        | False                         | Write response-length statistics for debugging                                         |
| `dump_only_message_path` | None                          | Export final messages without requesting the model                                     |

## Relationship to Parallel Inferencers

`ParallelGenInferencer`, `ParallelChatInferencer`, and `ParallelChatMLInferencer` are concurrent subclasses of their single-dataset counterparts. They fix `batch_size` to 1, use a thread pool to keep multiple requests from the **same dataset** in flight, write each sample incrementally to `tmp_*.jsonl` so a restart continues from completed samples, and report progress through `progress_tracker`.

The two layers are responsible for:

| Layer      | Component                     | Responsibility                                                                                        |
| ---------- | ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| Task       | `OpenICLInferConcurrentTask`  | Across datasets: one process manages several datasets, shares request budget, and writes status files |
| Inferencer | `ParallelGenInferencer`, etc. | Within a dataset: sample-level concurrent requests, incremental output, and progress callbacks        |

At runtime, the concurrent task **automatically** replaces `GenInferencer`/`ChatInferencer`/`ChatMLInferencer` in dataset configurations with the corresponding Parallel version and passes `max_infer_workers` plus a progress callback. Other inferencers such as `PPLInferencer` and `SCInferencer` are currently unsupported and raise an error.

A Parallel Inferencer can also be used **on its own** by declaring it directly in Dataset `infer_cfg`, without the concurrent task:

```python
infer_cfg = dict(
    # ...
    inferencer=dict(type=ParallelGenInferencer, max_infer_workers=16),
)
```

On its own, it provides only sample concurrency within one dataset, with no cross-dataset shared semaphore or status reporting, so it cannot cooperate with evaluation watching.

## OpenICLEvalWatchTask

The watching task inherits `OpenICLEvalTask` and uses exactly the same scoring logic—Evaluator, postprocessing, and result output. Only the trigger timing differs: the ordinary evaluation task runs after all inference ends, while the watching task starts alongside inference and scores outputs as they complete.

Its workflow is:

1. At startup, collect model-dataset combinations whose results file does not yet exist for the corresponding prediction, then wait 10 seconds for the heartbeat file to appear.
2. Every `watch_interval` seconds (default 5), scan status files. A combination is scored only when **all of its shards** are `done`; a combination with `running`, `fail`, or missing shards keeps waiting.
3. The main process updates `<work_dir>/infer_heartbeat` every 5 seconds during inference. If the heartbeat is older than `heartbeat_timeout` seconds (default 60), the watcher treats inference as stopped, skips every remaining combination, and exits. It **does not** score incomplete prediction files.
4. Every `log_interval` seconds (default 30), print the number of remaining combinations.

The heartbeat timeout should cover gaps between inference-task scheduling or restart. Increase it on a high-latency network filesystem. A dataset with status `fail` likewise waits until timeout before being skipped; inspect its failure reason directly under `infer_status/`.

Constructor arguments belong under `eval.runner.task`:

| Argument            | Default | Description                                                         |
| ------------------- | ------- | ------------------------------------------------------------------- |
| `watch_interval`    | 5.0     | Status-file scan interval in seconds                                |
| `heartbeat_timeout` | 60.0    | Heartbeat timeout in seconds; skip remaining datasets after timeout |
| `log_interval`      | 30.0    | Log interval for the remaining task count while waiting             |

## Complete Configuration Example

The model declares request concurrency, while `infer` and `eval` use the two tasks:

```python
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalWatchTask, OpenICLInferConcurrentTask

models = [
    dict(
        # ... remaining API-model fields ...
        max_workers=16,  # Request concurrency shared by all datasets in one task
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
            watch_interval=5,       # Status-file scan interval in seconds
            heartbeat_timeout=60,   # Skip remaining datasets after timeout
            log_interval=30,        # Waiting log interval in seconds
        ),
    ),
)
```

Place `watch_interval`, `heartbeat_timeout`, and `log_interval` inside `eval.runner.task`. The default values of the concurrent inference task's same-named `poll_interval` and `log_interval` suit most cases.
