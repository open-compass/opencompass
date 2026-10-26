# MMEngine-Based Configuration Syntax

This page is a quick reference for OpenCompass configuration syntax. If you have not yet completed an evaluation, first read [Running a Complete Evaluation from a Configuration](config_based_evaluation.md).

## Basic Format

A configuration is a Python file that declares evaluation details through top-level variables. A minimal evaluation configuration contains two lists:

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
| `work_dir`   | Experiment output root; can also be overridden by `--work-dir`       |

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

After import, dictionaries in a list can be modified. If the same base configuration is also reused by another variable, deep-copy it first to avoid unintended changes:

```python
from copy import deepcopy

models = deepcopy(gpt6_models)
models[0]['query_per_second'] = 2
models[0]['max_workers'] = 16
```

## Configuration Objects and Registered Types

`type` can be an imported Python class object:

```python
from opencompass.summarizers import DefaultSummarizer

summarizer = dict(type=DefaultSummarizer)
```

For components managed by an MMEngine Registry, `type` can also be a registered-name string. For example:

```python
infer_cfg = dict(
    prompt_template=dict(
        type='RawPromptTemplate',
        messages=[dict(role='user', content='{question}')],
    ),
    retriever=dict(type='ZeroRetriever'),
    inferencer=dict(type='GenInferencer'),
)
```

When these components are built, the three names above are resolved in the prompt-template, Retriever, and Inferencer registries, respectively. OpenCompass registries define module locations and can automatically import built-in modules during lookup, so these classes normally do not need to be imported explicitly.

A string must exactly match the component's registered name. If a registration defines an alias, use the alias rather than the Python class name. A custom component's module must be imported so that its registration decorator has executed. Names cannot be mixed across registries, and all remaining configuration arguments must match the target constructor. Note that not every `type` field is managed by a Registry.

The repository currently has no single CLI command that lists every Registry. You can import the built-in modules for a Registry and inspect its registered names. For example, to list all Inferencers:

```bash
python -c "from opencompass.registry import ICL_INFERENCERS as R; R.import_from_location(); print('\n'.join(sorted(R.module_dict)))"
```

To check whether one name exists, call `get()` directly:

```bash
python -c "from opencompass.registry import ICL_INFERENCERS as R; print(R.get('GenInferencer'))"
```

Depending on the component category, replace `ICL_INFERENCERS` with `MODELS`, `LOAD_DATASET`, `RUNNERS`, `PARTITIONERS`, `TASKS`, `ICL_RETRIEVERS`, `ICL_PROMPT_TEMPLATES`, `ICL_EVALUATORS`, `TEXT_POSTPROCESSORS`, or `DICT_POSTPROCESSORS`.

## Configuration and CLI Precedence

There is no universal rule that CLI arguments always override configuration-file values. The current entry point handles several common cases as follows:

- `models`, `datasets`, and `summarizer` from a configuration file take precedence. CLI `--models`, `--datasets`, `--summarizer`, and the Hugging Face shortcuts are not merged with them.
- `--work-dir` overrides configuration `work_dir`. Without the option, the configuration value is retained; if neither is set, OpenCompass uses `outputs/default`.
- `--max-num-workers`, `--max-workers-per-gpu`, and `--retry` take effect only when no configuration file is supplied. `--max-num-workers` sets both Runner concurrency and `NumWorkerPartitioner.num_worker`.

## Parsing and Checking a Configuration

Use MMEngine to check syntax independently:

```bash
python -c "from mmengine.config import Config; Config.fromfile('my_eval.py')"
```

Use `--dry-run` to inspect the configuration after OpenCompass fills defaults and to check task partitioning:

```bash
opencompass my_eval.py --dry-run --config-verbose
```
