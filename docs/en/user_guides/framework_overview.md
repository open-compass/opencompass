# OpenCompass Workflow and Core Concepts

An OpenCompass evaluation launches a pipeline that can be partitioned, parallelized, resumed, and reused:

```text
experiment configuration
  ├─ models: models under evaluation
  ├─ datasets: samples, input construction, and evaluation method
  └─ infer / eval / summarizer: task orchestration and result presentation
          ↓
configuration parsing and task partitioning
          ↓
inference (written to predictions)
          ↓
evaluation (written to results)
          ↓
summarization (written to summary)
```

## Evaluation Configuration

The configuration is the single entry point for an experiment. A minimal configuration needs only `models` and `datasets`; OpenCompass fills in the default inference and evaluation tasks required for local execution. A formal experiment usually also specifies `work_dir`, `infer`, `eval`, or `summarizer` to fix its concurrency strategy and aggregation convention.

Configuration files use Python syntax and reuse repository model, dataset, and summarizer configurations through MMEngine `read_base()`. A configuration is not an arbitrary executable launch script: it declares experiment objects and strategy, while the `opencompass` command performs the actual scheduling.

## Models

A model configuration describes how to call the model and the resources required for execution. It includes the model backend, weights or endpoint, context length, maximum output length, concurrency, and model hyperparameters. Common entry points include OpenAI-compatible endpoints, vendor SDKs, and local Hugging Face, LMDeploy, vLLM, and multimodal model classes.

See [Model Integration](models.md).

## Datasets

In OpenCompass, a dataset configuration contains more than a data path. It usually also declares:

- `reader_cfg`: input fields, answer fields, and the data-splitting strategy.
- `infer_cfg`: prompt structure, few-shot configuration, and inference methods such as PPL or Gen.
- `eval_cfg`: model-output postprocessing and metric computation.

The same raw data can therefore have multiple configuration variants, for example with different few-shot settings, prompts, or evaluators. See [Dataset Configuration](datasets.md).

## Inference, Evaluation, and Summarization

When evaluation tasks run, a Partitioner divides “model × dataset” into parallelizable Tasks, a Runner determines whether and how those Tasks execute locally or in another cluster environment, and each Task performs the actual inference or evaluation. For example, the API configuration in the basic tutorial uses `OpenICLInferConcurrentTask` for efficient concurrent inference and `OpenICLEvalWatchTask` to evaluate results as they become available.

Inference and evaluation outputs are written to `predictions/` and `results/`, respectively. The Summarizer then organizes subset results into final summary files. Because each stage stores its artifacts separately, an interrupted run can use `--reuse` to continue from the missing stage, or use `--mode eval` and `--mode viz` to process existing artifacts one stage at a time. See [Reuse, Recovery, and Staged Execution](../execution/tasks_and_runners.md#task-recovery-artifact-reuse-and-evaluation-only-runs) and [Understanding Outputs and Result Summaries](results_and_summarizer.md).

## Work Directory and Reproducibility

By default, outputs are written to `outputs/default/<timestamp>/`. Give every experiment a stable `--work-dir` and retain its `configs/` snapshot. Model version, data version, dependency environment, random arguments, and Judge model all affect results; the final score alone is not a sufficient record.

This page provides only a high-level description of each step in the OpenCompass workflow. To launch a complete configuration-based evaluation, see [Running a Complete Evaluation from a Configuration](config_based_evaluation.md).
