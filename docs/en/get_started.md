# Installation

1. Use the following commands to set up the OpenCompass environment:

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
```

If you want to customize the PyTorch version or related CUDA version, please refer to the [official documentation](https://pytorch.org/get-started/locally/) to set up the PyTorch environment. Note that OpenCompass requires `pytorch>=1.13`.

2. Install OpenCompass:

```bash
git clone https://github.com/opencompass/opencompass
cd opencompass
pip install -e .
```

3. Install humaneval (Optional)

If you want to perform evaluations on the humaneval dataset, follow these steps.

```
git clone https://github.com/openai/human-eval.git
cd human-eval
pip install -r requirments.txt
pip install -e .
cd ..
```

Please read the comments in `human_eval/execution.py` **lines 48-57** to understand the potential risks of executing the model generation code. If you accept these risks, uncomment **line 58** to enable code execution evaluation.

# Quick Start

In this section, we will use the example of testing LLaMA-7B on SIQA and PIQA to familiarize you with some
basic features of OpenCompass. Before running, make sure you have installed OpenCompass and have GPU computing
resources that meet the minimum requirements for LLaMA-7B.

## Prepare the Dataset

Create a `data` folder in the repository directory and place the dataset files in the `data` folder.

## Prepare the Evaluation Configuration File

Create the following configuration file `configs/llama.py`:

```python
from mmengine.config import read_base

with read_base():
    # Read the required dataset configurations directly from the preset dataset configurations
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets

# Concatenate the datasets to be evaluated into the datasets field
datasets = [*piqa_datasets, *siqa_datasets]

# Evaluate models supported by HuggingFace's `AutoModelForCausalLM` using `HuggingFaceCausalLM`
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        # Initialization parameters for `HuggingFaceCausalLM`
        path='huggyllama/llama-7b',
        tokenizer_path='huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        # Common parameters for all models, not specific to HuggingFaceCausalLM's initialization parameters
        abbr='llama-7b',            # Model abbreviation for result display
        max_out_len=100,            # Maximum number of generated tokens
        batch_size=16,
        run_cfg=dict(num_gpus=1),   # Run configuration for specifying resource requirements
    )
]
```

## Start the Evaluation

First, we can start the task in **debug mode** to check for any exceptions in model loading, dataset reading, or incorrect cache usage.

```shell
python run.py configs/llama.py -w outputs/llama --debug
```

However, in `--debug` mode, tasks are executed sequentially. After confirming that everything is correct, you
can disable the `--debug` mode to fully utilize multiple GPUs.

```shell
python run.py configs/llama.py -w outputs/llama
```

Here are some parameters related to evaluation that can help you configure more efficient inference tasks based on your environment:

- `-w outputs/llama`: Directory to save evaluation logs and results.
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
- `--partition my_part`: Slurm cluster partition.
- `--retry 2`: Number of retries for failed tasks.

## Obtaining Evaluation Results

After the evaluation is complete, the evaluation results table will be printed as follows:

```text
dataset    version    metric    mode      llama-7b
---------  ---------  --------  ------  ----------
piqa       1cf9f0     accuracy  ppl          77.75
siqa       e78df3     accuracy  gen          36.08
```

Additionally, the text and CSV format result files will be saved in the `summary` folder of the result directory.
