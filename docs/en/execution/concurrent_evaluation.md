# Cross-Task Concurrent Inference and Evaluation Watching

This page introduces the standard OpenCompass task types, with an emphasis on task orchestration for large-scale API evaluation. The basic tutorial uses a **concurrent inference task** (`OpenICLInferConcurrentTask`) so that one process concurrently handles multiple datasets for the same model. It also uses an **evaluation watching task** (`OpenICLEvalWatchTask`) to begin evaluating completed datasets before the inference stage has finished in full. Used together, they reduce scheduling overhead and allow inference and evaluation to overlap, making them suitable for API evaluations with large request volumes.

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

## OpenICLInferConcurrentTask

`OpenICLInferConcurrentTask` concurrently processes the datasets for the same model assigned to one Task by the Partitioner. It is recommended to set `NumWorkerPartitioner.num_worker` to 1 so that all datasets for each model are assigned to one unsplit Task. `LocalRunner.max_num_workers` sets the maximum number of Tasks that can run at the same time. Use 1 for a single-model evaluation, or increase it as appropriate for multiple models after accounting for available resources and rate limits.

The internal workflow is:

1. **API-model restriction:** after constructing the model, it checks `model.is_api`; a local GPU model raises an error.
2. **Dataset scheduling:** a thread pool runs multiple datasets at once, up to `min(dataset count, 32)`, and starts the next dataset early when the remaining samples in running tasks fall below `2 × max_workers`, preventing request starvation.
3. **Status reporting:** each model-dataset combination maintains a JSON file under `infer_status/` (`<dataset>_0.json`, `<dataset>_1.json`, and so on for multiple shards). Status moves through pending → running → done/fail, and execution progress is continuously written for the watching task.
4. **Resume:** a model-dataset combination whose prediction file already exists is skipped.

`max_workers` first uses the same-named field from the model configuration. If absent, it defaults to `min(32, CPU core count + 4)`.

Concurrency controls and related arguments are:

| Argument        | Default                       | Configuration location and description                                                           |
| --------------- | ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `max_workers`   | `min(32, CPU core count + 4)` | Model configuration field that controls the request-concurrency semaphore shared by all datasets |
| `poll_interval` | 1.0                           | Top-level Task field specifying the dataset-scheduling poll interval in seconds                  |
| `log_interval`  | 30.0                          | Top-level Task field specifying the progress-log interval in seconds                             |

### Relationship to Parallel Inferencers

`ParallelGenInferencer`, `ParallelChatInferencer`, and `ParallelChatMLInferencer` are concurrent subclasses of their corresponding single-dataset Inferencers. They fix `batch_size` at 1, use a thread pool to process multiple sample requests from the **same dataset** concurrently, write each result incrementally to `tmp_*.jsonl` so that completed samples can be skipped after a restart, and report progress through `progress_tracker`.

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

## Inference and Evaluation Task Configuration Example

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

## Ordinary Inference and Evaluation Tasks

`OpenICLInferTask` processes the model-dataset combinations assigned to the same Task by the Partitioner in sequence. It is suitable for one-stop deployment and evaluation of local models. The Partitioner determines the combinations and their partitioning, while the Runner controls the execution environment and number of concurrent Tasks.

`OpenICLEvalTask` reads existing predictions and computes evaluation metrics. It uses the same scoring logic as `OpenICLEvalWatchTask`; the difference is that the ordinary task normally starts after inference has completed, whereas the watching task starts evaluation for each completed dataset according to inference progress.

The usual selections are:

- large-scale API evaluation: `OpenICLInferConcurrentTask` + `OpenICLEvalWatchTask`;
- local one-stop evaluation: `OpenICLInferTask` + `OpenICLEvalTask`.
