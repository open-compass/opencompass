# Configuration Syntax and Reuse

This page is a quick reference for OpenCompass configuration syntax. If you have not yet completed an evaluation, first read [Running a Complete Evaluation from a Configuration](config_based_evaluation.md).

## Basic Format

A configuration is a Python file that declares an experiment through top-level variables. A minimal evaluation configuration contains two lists:

```python
models = [dict(type=..., abbr='my-model', ...)]
datasets = [dict(type=..., abbr='my-dataset', ...)]
```

Common top-level fields are:

| Field        | Purpose                                                              |
| ------------ | -------------------------------------------------------------------- |
| `models`     | Model backend, path, generation arguments, and resource requirements |
| `datasets`   | Data loading, input construction, inference method, and evaluator    |
| `infer`      | Partitioner, Runner, and Task for inference                          |
| `eval`       | Partitioner, Runner, and Task for evaluation                         |
| `summarizer` | Metric grouping, display order, and aggregate score                  |
| `work_dir`   | Experiment output root; can be overridden by `--work-dir`            |

## Reusing Configurations with `read_base()`

OpenCompass uses MMEngine's pure-Python configuration inheritance. Imports must be placed inside the `read_base()` context:

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

For a configuration outside the repository, the full `opencompass.configs...` path is clearest. Configurations within `opencompass/configs` can also use relative imports.

## Combining Lists

Multiple models or datasets can be concatenated directly:

```python
models = gpt6_models + other_api_models
datasets = gsm8k_datasets + math_datasets
```

Use meaningful aliases at import time to avoid overwriting values when multiple modules export `models` or `datasets`.

## Overriding an Imported Configuration

After import, dictionaries in a list can be modified. If the same base configuration is also reused by another variable, deep-copy it first to avoid accidental coupling:

```python
from copy import deepcopy

models = deepcopy(gpt6_models)
models[0]['query_per_second'] = 2
models[0]['max_workers'] = 16
```

A common mistake is mutating a shared object and thereby changing another experiment group in the same file.

## Configuration Objects and Registered Types

`type` can be an imported Python class:

```python
from opencompass.summarizers import DefaultSummarizer

summarizer = dict(type=DefaultSummarizer)
```

After parsing, OpenCompass builds the corresponding component through a registry. Configuration arguments must match the component constructor; fields cannot be interchanged between models, datasets, Runners, and other components.

## Configuration and CLI Precedence

In general, the configuration declares the experiment while the CLI controls the current launch. Common overrides include:

- `--work-dir` overrides configuration `work_dir`.
- `--debug` enables Runner debug mode.
- `--max-num-workers` takes effect only when the CLI generates the default Runner; an explicitly configured field takes precedence.
- `--mode` and `--reuse` control execution stages and artifact reuse.

Run `opencompass --help` for all options in the installed version. Do not copy arguments from documentation for an old version directly into a new environment.

## Parsing and Inspecting a Configuration

Use MMEngine to check syntax independently:

```bash
python -c "from mmengine.config import Config; Config.fromfile('my_eval.py')"
```

Inspect the configuration after OpenCompass fills defaults, together with task partitioning:

```bash
opencompass my_eval.py --dry-run --config-verbose
```

Note: `--dry-run` still creates an experiment timestamp directory and saves the final configuration snapshot, but does not execute inference tasks.

## Maintenance Principles

- Configuration names should express key differences in model, dataset, prompt, and evaluation method.
- Formal evaluations should pin model revisions, data versions, and generation arguments.
- Never store keys in the configuration repository; prefer environment variables.
- Split a large configuration into model, dataset, summarizer, and experiment entry-point files instead of copying whole dictionaries.
- Parse after editing, then dry-run, then perform a small-sample trial.

See [Model Integration](models.md) for model fields, [Dataset Configuration](datasets.md) for dataset structure, and [RawPromptTemplate](../prompt/raw_prompt_template.md) as the preferred prompt reference.
