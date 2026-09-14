# Five-Minute Quick Start

This page demonstrates two entry points with the same model and dataset. A **configuration file** is recommended because it preserves and reuses evaluation settings and supports version control and result reproduction. The **CLI** is convenient for a quick trial. The example evaluates `gpt-6-astra` through the OpenAI Responses API on 64 GSM8K demo samples.

First complete [Installation and Environment Setup](installation.md), enter the OpenCompass repository root, and set the API key:

```bash
export OPENAI_API_KEY=<your-api-key>
```

This example needs no local GPU, but it must be able to access the OpenAI API. The first run may also download the dataset from the network.

## Path 1: Use a Configuration File

The repository already provides a `gpt-6-astra` model configuration and a GSM8K demo dataset. Create `quick_start_eval.py`:

```python
# quick_start_eval.py
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets
    from opencompass.configs.models.openai.gpt_6_astra import \
        models

datasets = gsm8k_datasets
```

First check whether the configuration parses and how tasks will be partitioned. `--dry-run` does not perform model inference:

```bash
opencompass quick_start_eval.py --dry-run
```

After verifying it, run the complete workflow:

```bash
opencompass quick_start_eval.py \
    --work-dir outputs/quick_start_config \
    --debug
```

`--debug` runs tasks sequentially in the current process and displays logs directly, which is useful for initial troubleshooting. Formal batch evaluations normally do not need it.

## Path 2: Use the CLI Directly

The same evaluation can be run without creating a configuration file. Model and dataset names come from `opencompass/configs/models` and `opencompass/configs/datasets`:

```bash
opencompass \
    --models gpt_6_astra \
    --datasets demo_gsm8k_chat_gen \
    --work-dir outputs/quick_start_cli \
    --debug
```

You can likewise add `--dry-run` before execution. To find configuration names, run:

```bash
python tools/list_configs.py gpt_6_astra gsm8k
```

The CLI is suitable for quick validation, but not for evaluations requiring fine-grained control. Prefer a configuration file when combining multiple models and datasets, changing model concurrency or concrete request arguments, or customizing the execution strategy.

## Viewing Results

Every run creates a timestamp directory under `--work-dir`, containing primarily:

- `configs/`: snapshot of the effective configuration for this run.
- `predictions/`: per-sample model outputs.
- `results/`: metrics and details computed by the evaluator.
- `summary/`: terminal tables and aggregated results such as CSV files.

If execution fails, first inspect logs in the timestamp directory, then see [FAQ](../faq/index.md) under “Other Documentation.” After completing this page, read [Workflow and Core Concepts](../user_guides/framework_overview.md), then learn a reproducible formal configuration workflow in [Running a Complete Evaluation from a Configuration](../user_guides/config_based_evaluation.md).
