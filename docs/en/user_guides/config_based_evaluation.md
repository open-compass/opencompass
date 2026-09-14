# Running a Complete Evaluation from a Configuration

This tutorial starts from an empty file and builds an evaluation configuration that can be parsed, tested, and reproduced. The final experiment calls gpt-6-astra through the OpenAI Responses API, evaluates 64 GSM8K demo samples, and demonstrates explicit control over task partitioning, runners, and result summarization.

## 1. Create the Configuration File

Create `my_eval.py` in the repository root. First use `read_base()` to import existing model and dataset configurations:

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

Both `models` and `datasets` must be lists. Importing the model variable under an alias makes a configuration combining multiple models easier to read. `demo_gsm8k_chat_gen` selects only 64 test samples and is suitable for environment validation, not for reporting a formal benchmark result.

- To change local weights, API, or inference backend, see [Model Integration](models.md).
- To choose a dataset and configuration variant, see [Dataset Configuration](datasets.md).
- To understand `read_base()`, overrides, and list composition, see [Configuration Syntax and Reuse](config.md).

## 2. Understand Input and Scoring in the Dataset

The imported dataset already contains `reader_cfg`, `infer_cfg`, and `eval_cfg`. They determine which fields are read, how model input is constructed, and how metrics are computed from output. Do not infer the evaluation method from the dataset name alone; inspect the concrete configuration file for a formal evaluation.

New dataset configurations should describe messages with [RawPromptTemplate](../prompt/raw_prompt_template.md). The traditional [PromptTemplate](../prompt/raw_prompt_template.md#prompttemplate-traditional-template) remains supported; model-side role mapping is handled by the [Model-Side Conversation Template Protocol](../prompt/meta_template.md).

## 3. First Run with the Default Strategy

The minimal configuration above is runnable. When `infer` and `eval` are absent, `opencompass` fills in local defaults. First inspect the final configuration and task partitioning:

```bash
opencompass my_eval.py --dry-run --config-verbose
```

`--dry-run` parses the configuration and constructs tasks without loading the model or running inference. It reveals import-path, field-name, and resource-declaration errors early.

Then execute the complete evaluation:

```bash
opencompass my_eval.py \
    --work-dir outputs/my_eval \
    --debug
```

Debug mode executes in the current process and displays logs directly. After validation, remove `--debug` and adjust local concurrency with `--max-num-workers` or model-configuration parameters.

## 4. Explicitly Fix Task Orchestration in the Configuration

For an experiment committed to version control or reproduced over time, write the default strategy explicitly in `my_eval.py`. Append:

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

This section describes execution and does not change scoring rules in the dataset:

- A Partitioner decides how to divide models and datasets into tasks.
- A Runner decides where and with what concurrency tasks execute.
- A Task decides whether to run inference or read predictions for evaluation.

Both `num_worker` and `max_num_workers` are intentionally set to 1 because one concurrent task takes ownership of every dataset for the same model and schedules requests within that process. `OpenICLInferConcurrentTask` supports API models only; local GPU models should use `OpenICLInferTask` and `OpenICLEvalTask`.

CLI `--slurm` or `--dlc` replaces the configured runner with the corresponding execution backend. See [Task Partitioning, Runners, and Task Types](../execution/tasks_and_runners.md) for the responsibilities and concurrent-resource settings of these components, [Cross-Task Concurrent Inference and Evaluation Watching](../execution/concurrent_evaluation.md) for API concurrent-task behavior, and [Model Integration](models.md) for model-side multi-GPU and tensor-parallel declarations.

## 5. Configure Result Summarization

When `summarizer` is absent, the default summarizer is used. To fix display order, grouping, or aggregate metrics, inherit an existing summary configuration or specify one explicitly:

```python
from opencompass.summarizers import DefaultSummarizer

summarizer = dict(type=DefaultSummarizer)
```

The Dataset's Evaluator produces raw metrics, while the Summarizer organizes and presents them. Do not conflate these responsibilities. See [Evaluation Metrics](metrics.md) for metric meanings.

## 6. Run, Recover, and Execute by Stage

For a formal run:

```bash
opencompass my_eval.py --work-dir outputs/my_eval
```

Every launch creates a timestamp directory under `outputs/my_eval/`. If a run is interrupted, reuse existing artifacts from the most recent experiment:

```bash
opencompass my_eval.py --work-dir outputs/my_eval --reuse
```

You can also specify the timestamp and execute only one stage:

```bash
opencompass my_eval.py \
    --work-dir outputs/my_eval \
    --reuse 20260903_120000 \
    --mode eval
```

`--mode eval` and `--mode viz` require `--reuse`; otherwise OpenCompass does not know which predictions or results to read. See [Task Recovery, Reuse, and Evaluation-Only Reruns](../execution/reuse_and_resume.md) for complete semantics and safety boundaries, and [Command-Line Reference](../execution/cli_reference.md) for all CLI options.

## 7. Inspect Artifacts

A run's timestamp directory contains the following. See [Understanding Outputs and Result Summarization](results_and_summarizer.md) for details of each directory:

```text
outputs/my_eval/<timestamp>/
├── configs/       # Effective configuration snapshot
├── logs/          # Task logs outside debug mode
├── predictions/   # Per-sample model outputs
├── results/       # Dataset evaluation results and details
└── summary/       # Summary tables, CSV files, and other exports
```

At minimum, acceptance checks should cover the configuration snapshot, failed-task logs, several per-sample inputs and outputs, and final metrics. A score without its configuration, model version, and data version is not a reproducible evaluation record.

## 8. Extend into a Formal Experiment

When extending this configuration, change only one category of variables at a time:

```python
models = model_group_a + model_group_b
datasets = objective_datasets + subjective_datasets
```

For new or customized datasets, read [Adding a Dataset](../extension/new_dataset.md) and [Quickly Evaluating Your Own Data](../extension/custom_dataset.md). For a new model backend, read [Adding a Model](../extension/new_model.md). Judge-based, mathematical, and code-execution evaluations require additional dependencies and security boundaries; do not run them by merely replacing a data path.
