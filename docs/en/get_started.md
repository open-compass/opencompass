# Installation

1. Set up the OpenCompass environment:

   ```bash
   conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
   conda activate opencompass
   ```

   If you want to customize the PyTorch version or related CUDA version, please refer to the [official documentation](https://pytorch.org/get-started/locally/) to set up the PyTorch environment. Note that OpenCompass requires `pytorch>=1.13`.

2. Install OpenCompass:

   ```bash
   git clone https://github.com/open-compass/opencompass.git
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

   You can find example configs in `configs/models`. ([example](https://github.com/open-compass/opencompass/blob/eb4822a94d624a4e16db03adeb7a59bbd10c2012/configs/models/llama2_7b_chat.py))

   </details>

# Dataset Preparation

The datasets supported by OpenCompass mainly include two parts:

1. Huggingface datasets: The [Huggingface Datasets](https://huggingface.co/datasets) provide a large number of datasets, which will **automatically download** when running with this option.
2. Custom dataset: OpenCompass also provides some Chinese custom **self-built** datasets. Please run the following command to **manually download and extract** them.

Run the following commands to download and place the datasets in the `${OpenCompass}/data` directory can complete dataset preparation.

```bash
# Run in the OpenCompass directory
wget https://github.com/open-compass/opencompass/releases/download/0.1.1/OpenCompassData.zip
unzip OpenCompassData.zip
```

OpenCompass has supported most of the datasets commonly used for performance comparison, please refer to `configs/dataset` for the specific list of supported datasets.

# Quick Start

We will demonstrate some basic features of OpenCompass through evaluating pretrained models [OPT-125M](https://huggingface.co/facebook/opt-125m) and [OPT-350M](https://huggingface.co/facebook/opt-350m) on both [SIQA](https://huggingface.co/datasets/social_i_qa) and [Winograd](https://huggingface.co/datasets/winogrande) benchmark tasks with their config file located at [configs/eval_demo.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_demo.py).

Before running this experiment, please make sure you have installed OpenCompass locally and it should run successfully under one _GTX-1660-6G_ GPU.
For larger parameterized models like Llama-7B, refer to other examples provided in the [configs directory](https://github.com/open-compass/opencompass/tree/main/configs).

## Configure an Evaluation Task

In OpenCompass, each evaluation task consists of the model to be evaluated and the dataset. The entry point for evaluation is `run.py`. Users can select the model and dataset to be tested either via command line or configuration files.

`````{tabs}

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

Some sample outputs are:

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

For HuggingFace models, users can set model parameters directly through the command line without additional configuration files. For instance, for the `facebook/opt-125m` model, you can evaluate it with the following command:

```bash
python run.py --datasets siqa_gen winograd_ppl \
--hf-path facebook/opt-125m \
--model-kwargs device_map='auto' \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--max-seq-len 2048 \
--max-out-len 100 \
--batch-size 128  \
--num-gpus 1
```

```{tip}
For all HuggingFace related parameters supported by `run.py`, please read [Initiating Evaluation Task](./user_guides/experimentation.md#launching-an-evaluation-task).
```


````

````{tab} Configuration File

In addition to configuring the experiment through the command line, OpenCompass also allows users to write the full configuration of the experiment in a configuration file and run it directly through `run.py`. This method of configuration allows users to easily modify experimental parameters, provides a more flexible configuration, and simplifies the run command. The configuration file is organized in Python format and must include the `datasets` and `models` fields.

The test configuration for this time is [configs/eval_demo.py](/configs/eval_demo.py). This configuration introduces the required dataset and model configurations through the [inheritance mechanism](./user_guides/config.md#inheritance-mechanism) and combines the `datasets` and `models` fields in the required format.

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

````

`````

The configuration file evaluation method is more concise. The following sections will use this method as an example to explain the other features.

## Run Evaluation

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

```{warning}
OpenCompass usually assumes network is available. If you encounter network issues or wish to run OpenCompass in an offline environment, please refer to [FAQ - Network - Q1](./faq.md#network) for solutions.
```

## Explanations

### Model list - `models`

OpenCompass provides a series of pre-defined model configurations under `configs/models`. Below is the configuration snippet related to [opt-350m](/configs/models/hf_opt_350m.py) (`configs/models/hf_opt_350m.py`):

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
       run_cfg=dict(num_gpus=1),     # The required GPU numbers for this model
    )
```

When using configurations, we can specify the relevant files through the command-line argument ``` --models`` or import the model configurations into the  ```models\` list in the configuration file using the inheritance mechanism.

If the HuggingFace model you want to test is not among them, you can also directly specify the related parameters in the command line.

```bash
python run.py \
--hf-path facebook/opt-350m \  # HuggingFace model path
--tokenizer-path facebook/opt-350m \  # HuggingFace tokenizer path (if the same as the model path, can be omitted)
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # Arguments to construct the tokenizer
--model-kwargs device_map='auto' \  # Arguments to construct the model
--max-seq-len 2048 \  # Maximum sequence length the model can accept
--max-out-len 100 \  # Maximum number of tokens to generate
--batch-size 64  \  # Batch size
--num-gpus 1  # Number of GPUs required to run the model
```

The pretrained models 'facebook/opt-350m' and 'facebook/opt-125m' will be automatically downloaded from HuggingFace during the first run.

```{note}
More information about model configuration can be found in [Prepare Models](./user_guides/models.md).
```

### Dataset list - `datasets`

The translation is:

Similar to models, dataset configuration files are provided under `configs/datasets`. Users can use `--datasets` in the command line or import related configurations in the configuration file via inheritance.

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

```{note}
You can find more information from [Dataset Preparation](./user_guides/dataset_prepare.md).
```

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
