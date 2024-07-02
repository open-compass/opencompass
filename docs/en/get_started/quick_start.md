# Quick Start

![image](https://github.com/open-compass/opencompass/assets/22607038/d063cae0-3297-4fd2-921a-366e0a24890b)

## Overview

OpenCompass provides a streamlined workflow for evaluating a model, which consists of the following stages: **Configure** -> **Inference** -> **Evaluation** -> **Visualization**.

**Configure**: This is your starting point. Here, you'll set up the entire evaluation process, choosing the model(s) and dataset(s) to assess. You also have the option to select an evaluation strategy, the computation backend, and define how you'd like the results displayed.

**Inference & Evaluation**: OpenCompass efficiently manages the heavy lifting, conducting parallel inference and evaluation on your chosen model(s) and dataset(s). The **Inference** phase is all about producing outputs from your datasets, whereas the **Evaluation** phase measures how well these outputs align with the gold standard answers. While this procedure is broken down into multiple "tasks" that run concurrently for greater efficiency, be aware that working with limited computational resources might introduce some unexpected overheads, and resulting in generally slower evaluation. To understand this issue and know how to solve it, check out [FAQ: Efficiency](faq.md#efficiency).

**Visualization**: Once the evaluation is done, OpenCompass collates the results into an easy-to-read table and saves them as both CSV and TXT files. If you need real-time updates, you can activate lark reporting and get immediate status reports in your Lark clients.

Coming up, we'll walk you through the basics of OpenCompass, showcasing evaluations of pretrained models [OPT-125M](https://huggingface.co/facebook/opt-125m) and [OPT-350M](https://huggingface.co/facebook/opt-350m) on the [SIQA](https://huggingface.co/datasets/social_i_qa) and [Winograd](https://huggingface.co/datasets/winograd_wsc) benchmark tasks. Their configuration files can be found at [configs/eval_demo.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_demo.py).

Before running this experiment, please make sure you have installed OpenCompass locally and it should run successfully under one _GTX-1660-6G_ GPU.
For larger parameterized models like Llama-7B, refer to other examples provided in the [configs directory](https://github.com/open-compass/opencompass/tree/main/configs).

## Configuring an Evaluation Task

In OpenCompass, each evaluation task consists of the model to be evaluated and the dataset. The entry point for evaluation is `run.py`. Users can select the model and dataset to be tested either via command line or configuration files.

`````{tabs}
````{tab} Command Line (Custom HF Model)

For HuggingFace models, users can set model parameters directly through the command line without additional configuration files. For instance, for the `facebook/opt-125m` model, you can evaluate it with the following command:

```bash
python run.py --datasets siqa_gen winograd_ppl \
--hf-type base \
--hf-path facebook/opt-125m
```

Note that in this way, OpenCompass only evaluates one model at a time, while other ways can evaluate multiple models at once.

```{caution}
`--hf-num-gpus` does not stand for the actual number of GPUs to use in evaluation, but the minimum required number of GPUs for this model. [More](faq.md#how-does-opencompass-allocate-gpus)
```

:::{dropdown} More detailed example
:animate: fade-in-slide-down
```bash
python run.py --datasets siqa_gen winograd_ppl \
--hf-type base \  # HuggingFace model type, base or chat
--hf-path facebook/opt-125m \  # HuggingFace model path
--tokenizer-path facebook/opt-125m \  # HuggingFace tokenizer path (if the same as the model path, can be omitted)
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # Arguments to construct the tokenizer
--model-kwargs device_map='auto' \  # Arguments to construct the model
--max-seq-len 2048 \  # Maximum sequence length the model can accept
--max-out-len 100 \  # Maximum number of tokens to generate
--min-out-len 100 \  # Minimum number of tokens to generate
--batch-size 64  \  # Batch size
--hf-num-gpus 1  # Number of GPUs required to run the model
```
```{seealso}
For all HuggingFace related parameters supported by `run.py`, please read [Launching Evaluation Task](../user_guides/experimentation.md#launching-an-evaluation-task).
```
:::

````
````{tab} Command Line

Users can combine the models and datasets they want to test using `--models` and `--datasets`.

```bash
python run.py --models hf_opt_125m hf_opt_350m --datasets siqa_gen winograd_ppl
```

The models and datasets are pre-stored in the form of configuration files in `configs/models` and `configs/datasets`. Users can view or filter the currently available model and dataset configurations using `tools/list_configs.py`.

```bash
# List all configurations
python tools/list_configs.py
# List all configurations related to llama and mmlu
python tools/list_configs.py llama mmlu
```

:::{dropdown} More about `list_configs`
:animate: fade-in-slide-down

Running `python tools/list_configs.py llama mmlu` gives the output like:

```text
+-----------------+-----------------------------------+
| Model           | Config Path                       |
|-----------------+-----------------------------------|
| hf_llama2_13b   | configs/models/hf_llama2_13b.py   |
| hf_llama2_70b   | configs/models/hf_llama2_70b.py   |
| ...             | ...                               |
+-----------------+-----------------------------------+
+-------------------+---------------------------------------------------+
| Dataset           | Config Path                                       |
|-------------------+---------------------------------------------------|
| cmmlu_gen         | configs/datasets/cmmlu/cmmlu_gen.py               |
| cmmlu_gen_ffe7c0  | configs/datasets/cmmlu/cmmlu_gen_ffe7c0.py        |
| ...               | ...                                               |
+-------------------+---------------------------------------------------+
```

Users can use the names in the first column as input parameters for `--models` and `--datasets` in `python run.py`. For datasets, the same name with different suffixes generally indicates that its prompts or evaluation methods are different.
:::

:::{dropdown} Model not on the list?
:animate: fade-in-slide-down

If you want to evaluate other models, please check out the "Command Line (Custom HF Model)" tab for the way to construct a custom HF model without a configuration file, or "Configuration File" tab to learn the general way to prepare your model configurations.

:::

````

````{tab} Configuration File

In addition to configuring the experiment through the command line, OpenCompass also allows users to write the full configuration of the experiment in a configuration file and run it directly through `run.py`. The configuration file is organized in Python format and must include the `datasets` and `models` fields.

The test configuration for this time is [configs/eval_demo.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_demo.py). This configuration introduces the required dataset and model configurations through the [inheritance mechanism](../user_guides/config.md#inheritance-mechanism) and combines the `datasets` and `models` fields in the required format.

```python
from mmengine.config import read_base

with read_base():
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .datasets.winograd.winograd_ppl import winograd_datasets
    from .models.opt.hf_opt_125m import opt125m
    from .models.opt.hf_opt_350m import opt350m

datasets = [*siqa_datasets, *winograd_datasets]
models = [opt125m, opt350m]
```

When running tasks, we just need to pass the path of the configuration file to `run.py`:

```bash
python run.py configs/eval_demo.py
```

:::{dropdown} More about `models`
:animate: fade-in-slide-down

OpenCompass provides a series of pre-defined model configurations under `configs/models`. Below is the configuration snippet related to [opt-350m](https://github.com/open-compass/opencompass/blob/main/configs/models/opt/hf_opt_350m.py) (`configs/models/opt/hf_opt_350m.py`):

```python
# Evaluate models supported by HuggingFace's `AutoModelForCausalLM` using `HuggingFaceBaseModel`
from opencompass.models import HuggingFaceBaseModel

models = [
    # OPT-350M
    dict(
        type=HuggingFaceBaseModel,
        # Initialization parameters for `HuggingFaceBaseModel`
        path='facebook/opt-350m',
        # Below are common parameters for all models, not specific to HuggingFaceBaseModel
        abbr='opt-350m-hf',         # Model abbreviation
        max_out_len=1024,           # Maximum number of generated tokens
        batch_size=32,              # Batch size
        run_cfg=dict(num_gpus=1),   # The required GPU numbers for this model
    )
]
```

When using configurations, we can specify the relevant files through the command-line argument ` --models` or import the model configurations into the  `models` list in the configuration file using the inheritance mechanism.

```{seealso}
More information about model configuration can be found in [Prepare Models](../user_guides/models.md).
```
:::

:::{dropdown} More about `datasets`
:animate: fade-in-slide-down

Similar to models, dataset configuration files are provided under `configs/datasets`. Users can use `--datasets` in the command line or import related configurations in the configuration file via inheritance

Below is a dataset-related configuration snippet from `configs/eval_demo.py`:

```python
from mmengine.config import read_base  # Use mmengine.read_base() to read the base configuration

with read_base():
    # Directly read the required dataset configurations from the preset dataset configurations
    from .datasets.winograd.winograd_ppl import winograd_datasets  # Read Winograd configuration, evaluated based on PPL (perplexity)
    from .datasets.siqa.siqa_gen import siqa_datasets  # Read SIQA configuration, evaluated based on generation

datasets = [*siqa_datasets, *winograd_datasets]       # The final config needs to contain the required evaluation dataset list 'datasets'
```

Dataset configurations are typically of two types: 'ppl' and 'gen', indicating the evaluation method used. Where `ppl` means discriminative evaluation and `gen` means generative evaluation.

Moreover, [configs/datasets/collections](https://github.com/open-compass/opencompass/blob/main/configs/datasets/collections) houses various dataset collections, making it convenient for comprehensive evaluations. OpenCompass often uses [`base_medium.py`](/configs/datasets/collections/base_medium.py) for full-scale model testing. To replicate results, simply import that file, for example:

```bash
python run.py --models hf_llama_7b --datasets base_medium
```

```{seealso}
You can find more information from [Dataset Preparation](../user_guides/datasets.md).
```
:::


````

`````

```{warning}
OpenCompass usually assumes network is available. If you encounter network issues or wish to run OpenCompass in an offline environment, please refer to [FAQ - Network - Q1](./faq.md#network) for solutions.
```

The following sections will use configuration-based method as an example to explain the other features.

## Launching Evaluation

Since OpenCompass launches evaluation processes in parallel by default, we can start the evaluation in `--debug` mode for the first run and check if there is any problem. In `--debug` mode, the tasks will be executed sequentially and output will be printed in real time.

```bash
python run.py configs/eval_demo.py -w outputs/demo --debug
```

The pretrained models 'facebook/opt-350m' and 'facebook/opt-125m' will be automatically downloaded from HuggingFace during the first run.
If everything is fine, you should see "Starting inference process" on screen:

```bash
[2023-07-12 18:23:55,076] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...
```

Then you can press `ctrl+c` to interrupt the program, and run the following command in normal mode:

```bash
python run.py configs/eval_demo.py -w outputs/demo
```

In normal mode, the evaluation tasks will be executed parallelly in the background, and their output will be redirected to the output directory `outputs/demo/{TIMESTAMP}`. The progress bar on the frontend only indicates the number of completed tasks, regardless of their success or failure. **Any backend task failures will only trigger a warning message in the terminal.**

:::{dropdown} More parameters in `run.py`
:animate: fade-in-slide-down
Here are some parameters related to evaluation that can help you configure more efficient inference tasks based on your environment:

- `-w outputs/demo`: Work directory to save evaluation logs and results. In this case, the experiment result will be saved to `outputs/demo/{TIMESTAMP}`.
- `-r`: Reuse existing inference results, and skip the finished tasks. If followed by a timestamp, the result under that timestamp in the workspace path will be reused; otherwise, the latest result in the specified workspace path will be reused.
- `--mode all`: Specify a specific stage of the task.
  - all: (Default) Perform a complete evaluation, including inference and evaluation.
  - infer: Perform inference on each dataset.
  - eval: Perform evaluation based on the inference results.
  - viz: Display evaluation results only.
- `--max-partition-size 2000`: Dataset partition size. Some datasets may be large, and using this parameter can split them into multiple sub-tasks to efficiently utilize resources. However, if the partition is too fine, the overall speed may be slower due to longer model loading times.
- `--max-num-workers 32`: Maximum number of parallel tasks. In distributed environments such as Slurm, this parameter specifies the maximum number of submitted tasks. In a local environment, it specifies the maximum number of tasks executed in parallel. Note that the actual number of parallel tasks depends on the available GPU resources and may not be equal to this number.

If you are not performing the evaluation on your local machine but using a Slurm cluster, you can specify the following parameters:

- `--slurm`: Submit tasks using Slurm on the cluster.
- `--partition(-p) my_part`: Slurm cluster partition.
- `--retry 2`: Number of retries for failed tasks.

```{seealso}
The entry also supports submitting tasks to Alibaba Deep Learning Center (DLC), and more customized evaluation strategies. Please refer to [Launching an Evaluation Task](../user_guides/experimentation.md#launching-an-evaluation-task) for details.
```

:::

## Visualizing Evaluation Results

After the evaluation is complete, the evaluation results table will be printed as follows:

```text
dataset    version    metric    mode      opt350m    opt125m
---------  ---------  --------  ------  ---------  ---------
siqa       e78df3     accuracy  gen         21.55      12.44
winograd   b6c7ed     accuracy  ppl         51.23      49.82
```

All run outputs will be directed to `outputs/demo/` directory with following structure:

```text
outputs/default/
├── 20200220_120000
├── 20230220_183030     # one experiment pre folder
│   ├── configs         # Dumped config files for record. Multiple configs may be kept if different experiments have been re-run on the same experiment folder
│   ├── logs            # log files for both inference and evaluation stages
│   │   ├── eval
│   │   └── infer
│   ├── predictions   # Prediction results for each task
│   ├── results       # Evaluation results for each task
│   └── summary       # Summarized evaluation results for a single experiment
├── ...
```

The summarization process can be further customized in configuration and output the averaged score of some benchmarks (MMLU, C-Eval, etc.).

More information about obtaining evaluation results can be found in [Results Summary](../user_guides/summarizer.md).

## Additional Tutorials

To learn more about using OpenCompass, explore the following tutorials:

- [Prepare Datasets](../user_guides/datasets.md)
- [Prepare Models](../user_guides/models.md)
- [Task Execution and Monitoring](../user_guides/experimentation.md)
- [Understand Prompts](../prompt/overview.md)
- [Results Summary](../user_guides/summarizer.md)
- [Learn about Config](../user_guides/config.md)
