# Installation

## Basic Installation

1. Prepare the OpenCompass runtime environment using Conda:

```conda create --name opencompass python=3.10 -y
   # conda create --name opencompass_lmdeploy python=3.10 -y

   conda activate opencompass
```

If you want to customize the PyTorch version or related CUDA version, please refer to the [official documentation](https://pytorch.org/get-started/locally/) to set up the PyTorch environment. Note that OpenCompass requires `pytorch>=1.13`.

2. Install OpenCompass:
   - pip Installation
   ```bash
   # For support of most datasets and models
   pip install -U opencompass

   # Complete installation (supports more datasets)
   # pip install "opencompass[full]"

   # API Testing (e.g., OpenAI, Qwen)
   # pip install "opencompass[api]"
   ```
   - Building from Source Code If you want to use the latest features of OpenCompass
   ```bash
   git clone https://github.com/open-compass/opencompass opencompass
   cd opencompass
   pip install -e .
   ```

## Other Installations

### Inference Backends

```bash
 # Model inference backends. Since these backends often have dependency conflicts,
 # we recommend using separate virtual environments to manage them.
 pip install "opencompass[lmdeploy]"
 # pip install "opencompass[vllm]"
```

- LMDeploy

You can check if the inference backend has been installed successfully with the following command. For more information, refer to the [official documentation](https://lmdeploy.readthedocs.io/en/latest/get_started.html)

```bash
lmdeploy chat internlm/internlm2_5-1_8b-chat --backend turbomind
```

- vLLM

You can check if the inference backend has been installed successfully with the following command. For more information, refer to the [official documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)

```bash
vllm serve facebook/opt-125m
```

### API

OpenCompass supports different commercial model API calls, which you can install via pip or by referring to the [API dependencies](https://github.com/open-compass/opencompass/blob/main/requirements/api.txt) for specific API model dependencies.

```bash
pip install "opencompass[api]"

# pip install openai # GPT-3.5-Turbo / GPT-4-Turbo / GPT-4 / GPT-4o (API)
# pip install anthropic # Claude (API)
# pip install dashscope # Qwen (API)
# pip install volcengine-python-sdk # ByteDance Volcano Engine (API)
# ...
```

### Datasets

The basic installation supports most fundamental datasets. For certain datasets (e.g., Alpaca-eval, Longbench, etc.), additional dependencies need to be installed.

You can install these through pip or refer to the [additional dependencies](<(https://github.com/open-compass/opencompass/blob/main/requirements/extra.txt)>) for specific dependencies.

```bash
pip install "opencompass[full]"
```

For HumanEvalX / HumanEval+ / MBPP+, you need to manually clone the Git repository and install it.

```bash
git clone --recurse-submodules git@github.com:open-compass/human-eval.git
cd human-eval
pip install -e .
pip install -e evalplus
```

Some agent evaluations require installing numerous dependencies, which may conflict with existing runtime environments. We recommend creating separate conda environments to manage these.

```bash
# T-Eval
pip install lagent==0.1.2
# CIBench
pip install -r requirements/agent.txt
```

# Dataset Preparation

The datasets supported by OpenCompass mainly include three parts:

1. Huggingface datasets: The [Huggingface Datasets](https://huggingface.co/datasets) provide a large number of datasets, which will **automatically download** when running with this option.
   Translate the paragraph into English:

2. ModelScope Datasets: [ModelScope OpenCompass Dataset](https://modelscope.cn/organization/opencompass) supports automatic downloading of datasets from ModelScope.

   To enable this feature, set the environment variable: `export DATASET_SOURCE=ModelScope`. The available datasets include (sourced from OpenCompassData-core.zip):

   ```plain
   humaneval, triviaqa, commonsenseqa, tydiqa, strategyqa, cmmlu, lambada, piqa, ceval, math, LCSTS, Xsum, winogrande, openbookqa, AGIEval, gsm8k, nq, race, siqa, mbpp, mmlu, hellaswag, ARC, BBH, xstory_cloze, summedits, GAOKAO-BENCH, OCNLI, cmnli
   ```

3. Custom dataset: OpenCompass also provides some Chinese custom **self-built** datasets. Please run the following command to **manually download and extract** them.

Run the following commands to download and place the datasets in the `${OpenCompass}/data` directory can complete dataset preparation.

```bash
# Run in the OpenCompass directory
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

If you need to use the more comprehensive dataset (~500M) provided by OpenCompass, You can download and `unzip` it using the following command:

```bash
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
unzip OpenCompassData-complete-20240207.zip
cd ./data
find . -name "*.zip" -exec unzip "{}" \;
```

The list of datasets included in both `.zip` can be found [here](https://github.com/open-compass/opencompass/releases/tag/0.2.2.rc1)

OpenCompass has supported most of the datasets commonly used for performance comparison, please refer to `configs/dataset` for the specific list of supported datasets.

For next step, please read [Quick Start](./quick_start.md).
