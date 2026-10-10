# Quick Start

This page demonstrates two entry points with the same model and dataset. A **configuration file** is recommended because it preserves and reuses evaluation settings and supports version control and reproducibility, while the **CLI** is convenient for a quick trial. The example evaluates `gpt-6-astra` through the OpenAI Responses API on 64 GSM8K demonstration samples.

First complete [Installation and Environment Setup](installation.md) and enter the OpenCompass repository root. Then open `opencompass/configs/models/openai/gpt_6_astra.py` and configure the API key in either of the following ways.

For a temporary local test, pass the key directly in the model configuration:

```python
from opencompass.models import OpenAISDKResponse

models = [
    dict(
        type=OpenAISDKResponse,
        abbr='gpt-6-astra-response',
        path='gpt-6-astra',
        key='your-api-key',
        # Keep the remaining parameters unchanged.
    )
]
```

Prefer storing the key in a custom environment variable. For example, first set `OPENCOMPASS_API_KEY` in the shell:

```bash
export OPENCOMPASS_API_KEY="your-api-key"
```

Then read that environment variable in the model configuration:

```python
import os
from opencompass.models import OpenAISDKResponse

models = [
    dict(
        type=OpenAISDKResponse,
        abbr='gpt-6-astra-response',
        path='gpt-6-astra',
        key=os.getenv('OPENCOMPASS_API_KEY'),
        # Keep the remaining parameters unchanged.
    )
]
```

This example does not require a local GPU, but the current environment and account credentials must be able to access the corresponding API service. The first run may also download the dataset from the network.

## Path 1: Use a Configuration File

The repository already provides a `gpt-6-astra` model configuration and a GSM8K demo dataset. Create `quick_start_eval.py`:

```python
# quick_start_eval.py
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets
    from opencompass.configs.models.openai.gpt_6_astra import \
        models as gpt6_astra_models

datasets = gsm8k_datasets
models = gpt6_astra_models
```

First use `--dry-run` to check whether the configuration parses and how inference tasks will be partitioned. This option does not perform model inference:

```bash
opencompass quick_start_eval.py --dry-run
```

After verifying the configuration, run the complete workflow:

```bash
opencompass quick_start_eval.py \
    --work-dir outputs/quick_start_config \
    --debug
```

`--debug` runs tasks sequentially in the current process and prints logs directly to the terminal, which is useful for initial troubleshooting. Formal batch evaluations normally do not need it.

## Path 2: Use the CLI Directly

The same evaluation can be run without creating a configuration file. Obtain configuration filenames, without the `.py` suffix, from `opencompass/configs/models` and `opencompass/configs/datasets`, then pass them to `--models` and `--datasets`:

```bash
opencompass \
    --models gpt_6_astra \
    --datasets demo_gsm8k_chat_gen \
    --work-dir outputs/quick_start_cli \
    --debug
```

To find configuration names, run:

```bash
python tools/list_configs.py gpt_6_astra gsm8k
```

The CLI is suitable for quick validation, but not for evaluations requiring fine-grained control. Prefer a configuration file when combining multiple models and datasets, changing model concurrency or specific request parameters, or customizing the execution strategy.

## Viewing Results

Every run creates a timestamp directory under `--work-dir`, containing primarily:

- `configs/`: snapshot of the effective configuration for this run.
- `predictions/`: per-sample model outputs.
- `results/`: metrics and details computed by the evaluator.
- `summary/`: final summaries in formats including CSV and Markdown.

If execution fails, first inspect the terminal output or logs in the timestamp directory, then see [FAQ](../faq/index.md) under “Other Documentation.” After completing this page, read [Workflow and Core Concepts](../user_guides/framework_overview.md), then learn a reproducible configuration workflow in [Running a Complete Evaluation from a Configuration](../user_guides/config_based_evaluation.md).
