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

   If you want to **evaluate your models coding ability on the humaneval dataset**, follow this step.

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

4. Install Llama (Optional)

   If you want to **evaluate Llama / Llama-2 / Llama-2-chat with its official implementation**, follow this step.

   <details>
   <summary><b>click to show the details</b></summary>

   ```bash
   git clone https://github.com/facebookresearch/llama.git
   cd llama
   pip install -r requirements.txt
   pip install -e .
   cd ..
   ```

   You can find example configs in `configs/models`. ([example](https://github.com/InternLM/opencompass/blob/eb4822a94d624a4e16db03adeb7a59bbd10c2012/configs/models/llama2_7b_chat.py))

   </details>

# Dataset Preparation

The datasets supported by OpenCompass mainly include two parts:

1. Huggingface datasets: The [Huggingface Datasets](https://huggingface.co/datasets) provide a large number of datasets, which will **automatically download** when running with this option.
2. Custom dataset: OpenCompass also provides some Chinese custom **self-built** datasets. Please run the following command to **manually download and extract** them.

Run the following commands to download and place the datasets in the `${OpenCompass}/data` directory can complete dataset preparation.

```bash
# Run in the OpenCompass directory
wget https://github.com/InternLM/opencompass/releases/download/0.1.1/OpenCompassData.zip
unzip OpenCompassData.zip
```

OpenCompass has supported most of the datasets commonly used for performance comparison, please refer to `configs/dataset` for the specific list of supported datasets.

# Quick Start

The evaluation of OpenCompass relies on configuration files which must contain fields **`datasets`** and **`models`**.
The configurations specify the models and datasets to evaluate using **"run.py"**.

We will demonstrate some basic features of OpenCompass through evaluating pretrained models [OPT-125M](https://huggingface.co/facebook/opt-125m) and [OPT-350M](https://huggingface.co/facebook/opt-350m) on both [SIQA](https://huggingface.co/datasets/social_i_qa) and [Winograd](https://huggingface.co/datasets/winogrande) benchmark tasks with their config file located at [configs/eval_demo.py](https://github.com/InternLM/opencompass/blob/main/configs/eval_demo.py).

Before running this experiment, please make sure you have installed OpenCompass locally and it should run successfully under one _GTX-1660-6G_ GPU.
For larger parameterized models like Llama-7B, refer to other examples provided in the [configs directory](https://github.com/InternLM/opencompass/tree/main/configs).

Since OpenCompass launches evaluation processes in parallel by default, we can start the evaluation for the first run and check if there is any prblem. In debugging mode, the tasks will be executed sequentially and the status will be printed in real time.

```bash
python run.py configs/eval_demo.py -w outputs/demo --debug
```

If everything is fine, you should see "Starting inference process" on screen:

```bash
[2023-07-12 18:23:55,076] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...
```

Then you can press `ctrl+c` to interrupt the program, and then run the following command to start the parallel evaluation:

```bash
python run.py configs/eval_demo.py -w outputs/demo
```

Now let's go over the configuration file and the launch options used in this case.

## Explanations

### Dataset list - `datasets`

Below is the configuration snippet related to datasets in `configs/eval_demo.py`:

```python
from mmengine.config import read_base  # Use mmengine.read_base() to load base configs

with read_base():
    # Read the required dataset configurations directly from the preset dataset configurations
    from .datasets.winograd.winograd_ppl import winograd_datasets   # Load Winograd's configuration, which uses perplexity-based inference
    from .datasets.siqa.siqa_gen import siqa_datasets               # Load SIQA's configuration, which uses generation-based inference

datasets = [*siqa_datasets, *winograd_datasets]   # Concatenate the datasets to be evaluated into the datasets field
```

Various dataset configurations are available in [configs/datasets](https://github.com/InternLM/OpenCompass/blob/main/configs/datasets).
Some datasets have two types of configuration files within their folders named `ppl` and `gen`, representing different evaluation methods. Specifically, `ppl` represents discriminative evaluation, while `gen` stands for generative evaluation.

[configs/datasets/collections](https://github.com/InternLM/OpenCompass/blob/main/configs/datasets/collections) contains various collections of datasets for comprehensive evaluation purposes.

You can find more information from [Dataset Preparation](./user_guides/dataset_prepare.md).

### Model list - `models`

OpenCompass supports directly specifying the list of models to be tested in the configuration. For HuggingFace models, users usually do not need to modify the code. The following is the relevant configuration snippet:

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
       # Below are common parameters for all models, not specific to HuggingFaceCausalLM
       abbr='opt350m',               # Model abbreviation for result display
       max_seq_len=2048,             # The maximum length of the entire sequence
       max_out_len=100,              # Maximum number of generated tokens
       batch_size=64,                # batchsize
       run_cfg=dict(num_gpus=1),     # Run configuration for specifying resource requirements
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
       # Below are common parameters for all models, not specific to HuggingFaceCausalLM
       abbr='opt125m',                # Model abbreviation for result display
       max_seq_len=2048,              # The maximum length of the entire sequence
       max_out_len=100,               # Maximum number of generated tokens
       batch_size=128,                # batchsize
       run_cfg=dict(num_gpus=1),      # Run configuration for specifying resource requirements
    )

models = [opt350m, opt125m]
```

The pretrained models 'facebook/opt-350m' and 'facebook/opt-125m' will be automatically downloaded from HuggingFace during the first run.

More information about model configuration can be found in [Prepare Models](./user_guides/models.md).

### Launch Evaluation

When the config file is ready, we can start the task in **debug mode** to check for any exceptions in model loading, dataset reading, or incorrect cache usage.

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

## Obtaining Evaluation Results

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

More information about obtaining evaluation results can be found in [Results Summary](./user_guides/summarizer.md).

## Additional Tutorials

To learn more about using OpenCompass, explore the following tutorials:

- [Prepare Datasets](./user_guides/dataset_prepare.md)
- [Prepare Models](./user_guides/models.md)
- [Task Execution and Monitoring](./user_guides/experimentation.md)
- [Understand Prompts](./prompt/overview.md)
- [Results Summary](./user_guides/summarizer.md)
- [Learn about Config](./user_guides/config.md)
