# Accelerate Evaluation Inference with vLLM, LMDeploy, SGLang, or OpenAI

## Background

During the OpenCompass evaluation process, the Huggingface transformers library is used for inference by default. While this is a very general solution, there are scenarios where more efficient inference methods are needed to speed up the process, such as leveraging VLLM, LMDeploy, SGLang, or OpenAI-compatible APIs.

- [LMDeploy](https://github.com/InternLM/lmdeploy) is a toolkit designed for compressing, deploying, and serving large language models (LLMs), developed by the [MMRazor](https://github.com/open-mmlab/mmrazor) and [MMDeploy](https://github.com/open-mmlab/mmdeploy) teams.
- [vLLM](https://github.com/vllm-project/vllm) is a fast and user-friendly library for LLM inference and serving, featuring advanced serving throughput, efficient PagedAttention memory management, continuous batching of requests, fast model execution via CUDA/HIP graphs, quantization techniques (e.g., GPTQ, AWQ, SqueezeLLM, FP8 KV Cache), and optimized CUDA kernels.
- [SGLang](https://github.com/sgl-project/sglang) is a structured generation language designed for large language models (LLMs). It makes your interaction with models faster and more controllable.
- **OpenAI-compatible APIs** allow you to use any OpenAI-compatible endpoint for model inference, including official OpenAI models or self-hosted models with OpenAI-compatible API interfaces.

## Preparation for Acceleration

First, check whether the model you want to evaluate supports inference acceleration using vLLM, LMDeploy, SGLang, or OpenAI-compatible APIs. Additionally, ensure you have installed the required backend as per their official documentation. Below are the installation methods for reference:

### LMDeploy Installation Method

Install LMDeploy using pip (Python 3.8+) or from [source](https://github.com/InternLM/lmdeploy/blob/main/docs/en/build.md):

```bash
pip install lmdeploy
```

### VLLM Installation Method

Install vLLM using pip or from [source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source):

```bash
pip install vllm
```

### SGLang Installation Method

Install SGLang using pip or from [source](https://github.com/sgl-project/sglang):

```bash
pip install sglang
```

### OpenAI API Setup

For OpenAI-compatible APIs, you only need to install the openai package:

```bash
pip install openai
```

## Accelerated Evaluation Using VLLM, LMDeploy, SGLang, or OpenAI

### Method 1: Using Command Line Parameters to Change the Inference Backend

OpenCompass offers one-click evaluation acceleration. During evaluation, it can automatically convert Huggingface transformer models to VLLM, LMDeploy, SGLang, or OpenAI models for use. Below is an example code for evaluating the GSM8k dataset using the default Huggingface version of the llama3-8b-instruct model:

```python
# eval_gsm8k.py
from mmengine.config import read_base

with read_base():
    # Select a dataset list
    from .datasets.gsm8k.gsm8k_0shot_gen_a58960 import gsm8k_datasets as datasets
    # Select an interested model
    from ..models.hf_llama.hf_llama3_8b_instruct import models
```

Here, `hf_llama3_8b_instruct` specifies the original Huggingface model configuration, as shown below:

```python
from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-8b-instruct-hf',
        path='meta-llama/Meta-Llama-3-8B-Instruct',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]
```

To evaluate the GSM8k dataset using the default Huggingface version of the llama3-8b-instruct model, use:

```bash
python run.py config/eval_gsm8k.py
```

To accelerate the evaluation using vLLM, LMDeploy, SGLang, or OpenAI, you can use the following script:

**Using vLLM:**
```bash
python run.py config/eval_gsm8k.py -a vllm
```

**Using LMDeploy:**
```bash
python run.py config/eval_gsm8k.py -a lmdeploy
```

**Using SGLang:**
```bash
python run.py config/eval_gsm8k.py -a sglang
```

**Using OpenAI API:**
```bash
python run.py config/eval_gsm8k.py -a openai
```

Note: For OpenAI backend, you may need to configure additional parameters in your model config, such as `openai_api_base` and `api_key`.

### Method 2: Accelerating Evaluation via Deployed Inference Acceleration Service API

OpenCompass also supports accelerating evaluation by deploying vLLM, LMDeploy, or SGLang inference acceleration service APIs. Follow these steps:

1. Install the openai package:

```bash
pip install openai
```

2. Deploy the inference acceleration service API for vLLM or LMDeploy. Below is an example for LMDeploy:

```bash
lmdeploy serve api_server meta-llama/Meta-Llama-3-8B-Instruct --model-name Meta-Llama-3-8B-Instruct --server-port 23333
```

Parameters for starting the api_server can be checked using `lmdeploy serve api_server -h`, such as --tp for tensor parallelism, --session-len for the maximum context window length, --cache-max-entry-count for adjusting the k/v cache memory usage ratio, etc.

3. Once the service is successfully deployed, modify the evaluation script by changing the model configuration path to the service address, as shown below:

```python
from opencompass.models import OpenAISDK

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        abbr='Meta-Llama-3-8B-Instruct-LMDeploy-API',
        type=OpenAISDK,
        key='EMPTY', # API key
        openai_api_base='http://0.0.0.0:23333/v1',  # Service address
        path='Meta-Llama-3-8B-Instruct',  # Model name for service request
        tokenizer_path='meta-llama/Meta-Llama-3.1-8B-Instruct', # The tokenizer name or path, if set to `None`, uses the default `gpt-4` tokenizer
        rpm_verbose=True,  # Whether to print request rate
        meta_template=api_meta_template,  # Service request template
        query_per_second=1,  # Service request rate
        max_out_len=1024,  # Maximum output length
        max_seq_len=4096,  # Maximum input length
        temperature=0.01,  # Generation temperature
        batch_size=8,  # Batch size
        retry=3,  # Number of retries
    )
]
```

## Acceleration Effect and Performance Comparison

Below is a comparison table of the acceleration effect and performance when using VLLM or LMDeploy on a single A800 GPU for evaluating the Llama-3-8B-Instruct model on the GSM8k dataset:

| Inference Backend | Accuracy | Inference Time (minutes:seconds) | Speedup (relative to Huggingface) |
| ----------------- | -------- | -------------------------------- | --------------------------------- |
| Huggingface       | 74.22    | 24:26                            | 1.0                               |
| LMDeploy          | 73.69    | 11:15                            | 2.2                               |
| VLLM              | 72.63    | 07:52                            | 3.1                               |
