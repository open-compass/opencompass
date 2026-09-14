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

A model configuration describes how to call the model and the resources needed by one instance. It includes the backend, weights or endpoint, context length, maximum output length, batch size, generation arguments, and `run_cfg`. Common entry points include OpenAI-compatible endpoints, vendor SDKs, and local Hugging Face, LMDeploy, vLLM, and multimodal model classes. API models normally declare `run_cfg.num_gpus=0` and consume no local GPU.

See [Model Integration](models.md).

## Datasets

In OpenCompass, a dataset configuration contains more than a data path. It usually also declares:

- `reader_cfg`: input fields, answer fields, and data splits.
- `infer_cfg`: prompts, example retrievers, and generation or PPL inferencers.
- `eval_cfg`: answer postprocessing and metric computation.

The same raw data can therefore have multiple configuration variants, for example with different few-shot settings, prompts, or evaluators. See [Dataset Configuration](datasets.md).

## Inference, Evaluation, and Summarization

During inference, a Partitioner divides “model × dataset” into tasks, a Runner determines how tasks execute locally or in another cluster environment, and a Task performs the actual inference. The API example in the basic tutorial uses `OpenICLInferConcurrentTask` for concurrent inference together with `OpenICLEvalWatchTask` for evaluation as outputs complete; local models still use the ordinary inference and evaluation tasks. Outputs are written to `predictions/`.

The evaluation stage reads predictions, uses the Dataset's Evaluator to compute scores, and writes them to `results/`. A Summarizer then organizes subset results into terminal tables and summary files. Because inference and evaluation results are stored separately, `--reuse` can rerun only a missing stage, while `--mode eval` and `--mode viz` can process existing outputs.

## Work Directory and Reproducibility

By default, outputs are written to `outputs/default/<timestamp>/`. Give every experiment a stable `--work-dir` and retain its `configs/` snapshot. Model version, data version, dependency environment, random arguments, and Judge model all affect results; the final score alone is not a sufficient record.

This page provides only a high-level description of each step in the OpenCompass workflow. To launch a complete configuration-based evaluation, see [Running a Complete Evaluation from a Configuration](config_based_evaluation.md).
