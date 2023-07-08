# Installation

1. Set up the OpenCompass environment:

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
```

If you want to customize the PyTorch version or related CUDA version, please refer to the [official documentation](https://pytorch.org/get-started/locally/) to set up the PyTorch environment. Note that OpenCompass requires `pytorch>=1.13`.

2. Install OpenCompass:

```bash
git clone https://github.com/InternLM/opencompass.git
cd opencompass
pip install -e .
```

3. Install humaneval (Optional)

If you want to **evaluate your models coding ability on the humaneval dataset**, execute this step otherwise skip it.

<details>
<summary><b>click to show the details</b></summary>

```bash
git clone https://github.com/openai/human-eval.git
cd human-eval
pip install -r requirements.txt
pip install -e .
cd ..
```

Please read the comments in `human_eval/execution.py` **lines 48-57** to understand the potential risks of executing the model generation code. If you accept these risks, uncomment **line 58** to enable code execution evaluation.

</details>

# Dataset Preparation

The datasets supported by OpenCompass mainly include two parts:

1. Huggingface datasets: The [Huggingface Datasets](https://huggingface.co/datasets) provide a large number of datasets, which will **automatically download** when running with this option.
2. Custom dataset: OpenCompass also provides some Chinese custom **self-built** datasets. Please run the following command to **manually download and extract** them.

Run the following commands to download and place the datasets in the '${OpenCompass}/data' directory can complete dataset preparation.

```bash
# Run in the OpenCompass directory
wget https://github.com/InternLM/opencompass/releases/download/0.1.0/OpenCompassData.zip
unzip OpenCompassData.zip
```

OpenCompass has supported most of the datasets commonly used for performance comparison, please refer to `configs/dataset` for the specific list of supported datasets.

# Quick Start

The evaluation of OpenCompass relies on configuration files which must contain fields **`datasets`** and **`models`**.
The configurations specify the models and datasets to evaluate using **"run.py"**.

We will demonstrate some basic features of OpenCompass through evaluating pretrained models [OPT-125M](<(https://huggingface.co/facebook/opt-125m)>) and [OPT-350M](https://huggingface.co/facebook/opt-350m) on both [SIQA](https://huggingface.co/datasets/social_i_qa) and [Winograd](https://huggingface.co/datasets/winogrande) benchmark tasks with their config file located at [configs/eval_demo.py](https://github.com/InternLM/opencompass/blob/main/configs/eval_demo.py).

Before running this experiment, please make sure you have installed OpenCompass locally and it should run successfully under one _GTX-1660-6G_ GPU.
For larger parameterized models like Llama-7B, refer to other examples provided in the [configs directory](https://github.com/InternLM/opencompass/tree/main/configs).

To start the evaluation task, use the following command:

```bash
python run.py configs/eval_demo.py --debug
```

While running the demo, let's go over the details of the configuration content and launch options used in this case.

## Step by step

<details>
<summary><b>Learn about `datasets`</b></summary>

```python
from mmengine.config import read_base

with read_base():
    # Read the required dataset configurations directly from the preset dataset configurations
    from .datasets.winograd.winograd_ppl import winograd_datasets   # ppl inference
    from .datasets.siqa.siqa_gen import siqa_datasets               # gen inference

datasets = [*siqa_datasets, *winograd_datasets]   # Concatenate the datasets to be evaluated into the datasets field
```

Various dataset configurations are available in [configs/datasets](https://github.com/InternLM/OpenCompass/blob/main/configs/datasets).
Some datasets have two types of configuration files within their folders named `'ppl'` and `'gen'`, representing different evaluation methods. Specifically, `'ppl'` represents discriminative evaluation, while `'gen'` stands for generative evaluation.

[configs/datasets/collections](https://github.com/InternLM/OpenCompass/blob/main/configs/datasets/collections) contains various collections of datasets for comprehensive evaluation purposes.

</details>

<details>
<summary><b>Learn about `models`</b></summary>

The pretrained models 'facebook/opt-350m' and 'facebook/opt-125m' from HuggingFace supports automatic downloading.

```python
# Evaluate models supported by HuggingFace's `AutoModelForCausalLM` using `HuggingFaceCausalLM`
from opencompass.models import HuggingFaceCausalLM

# OPT-350M
opt350m = dict(
       type=HuggingFaceCausalLM,
       # Initialization parameters for `HuggingFaceCausalLM`
       path='facebook/opt-350m',
       tokenizer_path='facebook/opt-350m',
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True),
       model_kwargs=dict(device_map='auto'),
       max_seq_len=2048,
       # Common parameters for all models, not specific to HuggingFaceCausalLM's initialization parameters
       abbr='opt350m',                    # Model abbreviation for result display
       max_out_len=100,                   # Maximum number of generated tokens
       batch_size=64,                     # batchsize
       run_cfg=dict(num_gpus=1),          # Run configuration for specifying resource requirements
    )

# OPT-125M
opt125m = dict(
       type=HuggingFaceCausalLM,
       # Initialization parameters for `HuggingFaceCausalLM`
       path='facebook/opt-125m',
       tokenizer_path='facebook/opt-125m',
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True),
       model_kwargs=dict(device_map='auto'),
       max_seq_len=2048,
       # Common parameters for all models, not specific to HuggingFaceCausalLM's initialization parameters
       abbr='opt125m',                # Model abbreviation for result display
       max_out_len=100,               # Maximum number of generated tokens
       batch_size=128,                # batchsize
       run_cfg=dict(num_gpus=1),      # Run configuration for specifying resource requirements
    )

models = [opt350m, opt125m]
```

</details>

<details>
<summary><b>Launch Evaluation</b></summary>

First, we can start the task in **debug mode** to check for any exceptions in model loading, dataset reading, or incorrect cache usage.

```shell
python run.py configs/eval_demo.py -w outputs/demo --debug
```

However, in `--debug` mode, tasks are executed sequentially. After confirming that everything is correct, you
can disable the `--debug` mode to fully utilize multiple GPUs.

```shell
python run.py configs/eval_demo.py -w outputs/demo
```

Here are some parameters related to evaluation that can help you configure more efficient inference tasks based on your environment:

- `-w outputs/demo`: Directory to save evaluation logs and results.
- `-r`: Restart the previous (interrupted) evaluation.
- `--mode all`: Specify a specific stage of the task.
  - all: Perform a complete evaluation, including inference and evaluation.
  - infer: Perform inference on each dataset.
  - eval: Perform evaluation based on the inference results.
  - viz: Display evaluation results only.
- `--max-partition-size 2000`: Dataset partition size. Some datasets may be large, and using this parameter can split them into multiple sub-tasks to efficiently utilize resources. However, if the partition is too fine, the overall speed may be slower due to longer model loading times.
- `--max-num-workers 32`: Maximum number of parallel tasks. In distributed environments such as Slurm, this parameter specifies the maximum number of submitted tasks. In a local environment, it specifies the maximum number of tasks executed in parallel. Note that the actual number of parallel tasks depends on the available GPU resources and may not be equal to this number.

If you are not performing the evaluation on your local machine but using a Slurm cluster, you can specify the following parameters:

- `--slurm`: Submit tasks using Slurm on the cluster.
- `--partition(-p) my_part`: Slurm cluster partition.
- `--retry 2`: Number of retries for failed tasks.

```{tip}
The entry also supports submitting tasks to Alibaba Deep Learning Center (DLC), and more customized evaluation strategies. Please refer to [Launching an Evaluation Task](./user_guides/experimentation.md#launching-an-evaluation-task) for details.
```

</details>

## Obtaining Evaluation Results

After the evaluation is complete, the evaluation results table will be printed as follows:

```text
dataset    version    metric    mode      opt350m    opt125m
---------  ---------  --------  ------  ---------  ---------
siqa       e78df3     accuracy  gen         21.55      12.44
winograd   b6c7ed     accuracy  ppl         51.23      49.82
```

All run outputs will default to `outputs/default/` directory with following structure:

```text
outputs/default/
├── 20200220_120000
├── 20230220_183030     # one experiment pre folder
│   ├── configs         # replicable config files
│   ├── logs            # log files for both inference and evaluation stages
│   │   ├── eval
│   │   └── infer
│   ├── predictions     # json format of per data point inference result
│   └── results         # numerical conclusions of each evaluation session
├── ...
```

Each timestamp folder represents one experiment with the following contents:

- `configs`: configuration file storage;
- `logs`: log file storage for both **inference** and **evaluation** stages;
- `predictions`: json format output of inference result per data points;
- `results`: json format output of numerical conclusion on each evaluation session.

## Additional Tutorials

To learn more about using OpenCompass, explore the following tutorials:

- [Preparing Datasets](./user\_guides/dataset\_prepare.md)
- [Customizing Models](./user\_guides/models.md)
- [Exploring Experimentation Workflows](./user\_guides/experimentation.md)
- [Understanding Prompts](./prompt/overview.md)
