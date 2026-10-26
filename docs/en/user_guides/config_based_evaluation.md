# Running a Complete Evaluation with a Configuration File

This tutorial builds a complete evaluation configuration from an empty file. The final experiment evaluates `gpt-6-astra` through the OpenAI Responses API on 64 GSM8K demonstration samples, explicitly controls task partitioning and execution, and summarizes the evaluation results.

## 1. Create the Configuration File

Create `my_eval.py` in the repository root. First, use `read_base()` to import the existing model and dataset configurations:

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

Both `models` and `datasets` must be lists. Importing model variables under aliases keeps combinations of multiple model configurations readable. `demo_gsm8k_chat_gen` selects only 64 test samples and is intended for environment verification; it does not represent a formal benchmark result.

- To change local model weights, an API, or an inference backend, see [Model Integration](models.md).
- To change a dataset or inspect its detailed configuration, see [Dataset Configuration](datasets.md).
- To understand `read_base()`, overrides, and variable composition, see [Configuration Syntax and Reuse](config.md).

## 2. Run Evaluation Tasks with the Default Strategy

The configuration above is already a valid minimal configuration for `opencompass`. If `infer` and `eval` are not specified, `opencompass` supplies local defaults. Use the following command to inspect the effective configuration and task partitioning:

```bash
opencompass my_eval.py --dry-run --config-verbose
```

Then run the complete evaluation:

```bash
opencompass my_eval.py \
    --work-dir outputs/my_eval \
    --debug
```

## 3. Configure the Execution Strategy

Set `infer` and `eval` to explicitly specify the task partitioning strategy, execution environment, and task type:

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

This section controls execution and does not change the dataset's scoring rules:

- The Partitioner determines how model-dataset combinations are divided into subtasks. `NumWorkerPartitioner` distributes the datasets under evaluation as evenly as possible among no more than `num_worker` subtasks. With `num_worker=1`, all datasets for the same model are assigned to one subtask.
- The Runner determines where tasks execute. `LocalRunner` starts tasks on the local machine, and `max_num_workers=1` allows this Runner to execute at most one task at a time. This argument controls task-level concurrency, not API-request concurrency within a task.
- A Task is the actual execution unit for inference or evaluation and defines the stage entry point and processing logic. `OpenICLInferConcurrentTask` performs inference and uses the model-side `max_workers` setting for concurrency, but it supports API models only. Local models loaded directly in the evaluation process must use `OpenICLInferTask`. `OpenICLEvalWatchTask` performs evaluation and uses monitoring and heartbeat mechanisms to begin scoring a dataset as soon as its inference completes; local models are commonly paired with `OpenICLEvalTask`.

For the responsibilities, primary parameters, and selection of these component types, see [Task Partitioning, Runners, and Task Types](../execution/tasks_and_runners.md) and [Cross-task Concurrent Inference and Evaluation Monitoring](../execution/concurrent_evaluation.md).

## 4. Configure Result Summaries

Use `summarizer` to select the content, metrics, and order in the final summary table:

```python
summarizer = dict(
    dataset_abbrs=[
        'All Results',
        ['demo_gsm8k', 'accuracy'],
    ],
)
```

Items in `dataset_abbrs` appear in list order. A plain string such as `All Results` is displayed as a text row, while `['demo_gsm8k', 'accuracy']` selects a dataset `abbr` and metric. The Summarizer only organizes and presents metrics produced by the Evaluator; it does not change the scoring rules.

## 5. Run, Resume, and Execute Individual Stages

Run the evaluation with the following command. You may also place `work_dir` in the configuration file for reuse:

```bash
opencompass my_eval.py --work-dir outputs/my_eval
```

Each invocation creates a timestamped directory under `outputs/my_eval/`. If a run is interrupted, reuse the existing artifacts from the latest experiment:

```bash
opencompass my_eval.py --work-dir outputs/my_eval --reuse
```

You can also specify a timestamp and execute only one stage:

```bash
opencompass my_eval.py \
    --work-dir outputs/my_eval \
    --reuse 20260903_120000 \
    --mode eval
```

`--mode eval` and `--mode viz` must be used with `--reuse`; otherwise, OpenCompass cannot determine which predictions or results to read. For reuse behavior, see [Task Recovery, Artifact Reuse, and Evaluation-only Runs](../execution/tasks_and_runners.md#task-recovery-artifact-reuse-and-evaluation-only-runs). For all command-line arguments, see the [CLI Reference](../execution/cli_reference.md).

## 6. Complete Configuration File

Combining the preceding sections produces the following complete `my_eval.py`:

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

A command-line `--work-dir` overrides `work_dir` in the configuration file. After saving the file, run `opencompass my_eval.py --dry-run --config-verbose` to inspect the effective configuration and task partitioning before starting the complete evaluation.

## 7. Inspect the Artifacts

Each timestamped run directory contains the following. For details about the structure of these artifacts, see [Understanding Outputs and Result Summaries](results_and_summarizer.md).

```text
outputs/my_eval/<timestamp>/
├── configs/       # Snapshot of the effective configuration
├── logs/          # Task logs when debug mode is disabled
├── predictions/   # Per-sample model outputs
├── results/       # Dataset evaluation results and details
└── summary/       # Summary tables, CSV files, and other formats
```

## 8. Extend the Evaluation Configuration

A configuration file may combine multiple models and evaluation sets, define a shared Judge model, and apply dataset-side context settings in batches. For example:

```python
# Combine model and dataset configurations imported above with read_base().
models = model_group_a + model_group_b
datasets = objective_datasets + subjective_datasets

# The Judge model is used only for scoring and is not part of models above.
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
    # Override the maximum output length for every dataset Inferencer.
    inferencer = dataset.get('infer_cfg', {}).get('inferencer')
    if inferencer is not None:
        inferencer['max_out_len'] = 128000

    # Override Evaluator configurations that use a Judge model directly.
    evaluator = dataset.get('eval_cfg', {}).get('evaluator', {})
    if 'judge_cfg' in evaluator:
        evaluator['judge_cfg'] = judge_cfg
    # Also support an LLM Evaluator nested inside a cascade Evaluator.
    if ('llm_evaluator' in evaluator
            and 'judge_cfg' in evaluator['llm_evaluator']):
        evaluator['llm_evaluator']['judge_cfg'] = judge_cfg
```
