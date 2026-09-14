# Installation and Environment Setup

OpenCompass requires Python 3.8 or later. Model backends impose their own version constraints on PyTorch, CUDA, and inference frameworks. When preparing a GPU environment, first install a PyTorch build compatible with the model and backend, then install OpenCompass.

## Creating an Isolated Environment

Python 3.12 is recommended:

```bash
conda create -n opencompass python=3.12 -y
conda activate opencompass
```

Both the regular and full OpenCompass installations support Python 3.12. However, code-execution evaluations such as APPS (`apps`, `apps_mini`), TACO, and LiveCodeBench Code Generation depend on `pyext==0.7`. That package is incompatible with Python 3.11 and later, so create a Python 3.10 environment when running these evaluations.

LMDeploy and vLLM may require different versions of PyTorch, CUDA, or other dependencies. If you need multiple inference backends, create a separate virtual environment for each backend.

## Choosing an Installation Method

For common language models and datasets, install the base package:

```bash
pip install -U opencompass
```

Install optional dependencies according to the capabilities you need:

```bash
pip install "opencompass[api]"       # OpenAI, Anthropic, and other API models
pip install "opencompass[full]"      # More datasets and evaluation dependencies
pip install "opencompass[vlm]"       # Multimodal evaluation
pip install "opencompass[lmdeploy]"  # LMDeploy backend
pip install "opencompass[vllm]"      # vLLM backend
```

To use the latest code or contribute to development, install from source:

```bash
git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -e .
```

A source installation also registers the `opencompass` command. From the repository root, the equivalent `python run.py` entry point is also available.

## Verifying the Installation

The following commands apply to both PyPI and source installations. `which python` confirms the active Python environment; the next command reports the OpenCompass version and actual import path; the final command verifies that the CLI entry point and its dependencies are available.

```bash
which python
python -c "import opencompass; print(opencompass.__version__); print(opencompass.__file__)"
opencompass --help
```

## Data Caches

Datasets are normally downloaded on first use. On a shared machine, configure cache directories through these environment variables:

```bash
export HF_DATASETS_CACHE=/path/to/huggingface-cache/datasets
export COMPASS_DATA_CACHE=/path/to/opencompass-data-cache
```

`HF_DATASETS_CACHE` manages the Hugging Face dataset cache, while `COMPASS_DATA_CACHE` specifies the OpenCompass data-cache root. Download and loading behavior differs by dataset. See [Data Sources, Caches, and Offline Operation](../user_guides/data_and_cache.md) for detailed rules and offline preparation.

## Checking Models and Inference Backends

Successfully installing LMDeploy or vLLM does not guarantee that the target model is compatible with that backend. First load a small model with the chosen backend, then run a demo dataset through OpenCompass. For an API model, also check the service URL, model name, key environment variable, rate limit, and timeout configuration.

## Start Evaluating

After installation, continue to the [Five-Minute Quick Start](quick_start.md). For dependency, memory, or download issues, see [FAQ and Troubleshooting](../faq/index.md).
