# Accelerating Inference Evaluation with VLLM or LMDeploy

## Background

In the evaluation process of OpenCompass, the default method is to use Huggingface's transformers library for inference, which is a very versatile solution. However, in some cases, we may require more efficient inference methods to speed up this process, such as leveraging VLLM or LMDeploy.

- [LMDeploy](https://github.com/InternLM/lmdeploy) is a toolkit for compressing, deploying, and serving large language models (LLMs), developed by the [MMRazor](https://github.com/open-mmlab/mmrazor) and [MMDeploy](https://github.com/open-mmlab/mmdeploy) teams.
- [vLLM](https://github.com/vllm-project/vllm) is a fast and easy-to-use LLM inference and serving library, featuring advanced serving throughput, efficient memory management with PagedAttention, continuous batching of requests, fast model execution with CUDA/HIP graphs, quantization techniques (such as GPTQ, AWQ, SqueezeLLM, FP8 KV Cache), and optimized CUDA kernels.

## Preparation for Acceleration

First, check if the model you want to evaluate supports inference acceleration using VLLM or LMDeploy. Then, ensure you have installed VLLM or LMDeploy. Here are the reference installation methods based on their official documentation:

### LMDeploy Installation

Install using pip (Python 3.8+) or from [source](https://github.com/InternLM/lmdeploy/blob/main/docs/en/build.md):

```bash
pip install lmdeploy
```

### VLLM Installation

Install using pip or from [source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source):

```bash
pip install vllm
```

## Using VLLM or LMDeploy for Evaluation

OpenCompass provides a one-click evaluation acceleration feature, which automatically converts Huggingface transformer models to VLLM or LMDeploy models during the evaluation process. Below is a script that evaluates the GSM8k dataset using the default Huggingface version of the Internlm2-chat-7b model:

### OpenCompass Main Repository

```python
# eval_gsm8k.py
from mmengine.config import read_base

with read_base():
    # choose a list of datasets
    from .datasets.gsm8k.gsm8k_0shot_gen_a58960 import gsm8k_datasets as datasets
    # choose a model of interest
    from ..models.hf_llama.hf_llama3_8b_instruct import models
```

Here, `hf_llama3_8b_instruct` is the original Huggingface model config, as follows:

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

To evaluate the Llama3-8b-instruct model on the GSM8k dataset using the default Huggingface version, run:

```bash
python run.py config/eval_gsm8k.py
```

To use VLLM or LMDeploy for accelerated evaluation, use the following scripts:

```bash
python run.py config/eval_gsm8k.py -a vllm
```

or

```bash
python run.py config/eval_gsm8k.py -a lmdeploy
```

## Performance Comparison

Below is a performance comparison table for evaluating the GSM8k dataset using VLLM or LMDeploy, with single A800 GPU:

| Inference Backend | Accuracy | Inference Time (min:sec) | Speedup (relative to Huggingface) |
| ----------------- | -------- | ------------------------ | --------------------------------- |
| Huggingface       | 74.22    | 24:26                    | 1.0                               |
| LMDeploy          | 73.69    | 11:15                    | 2.2                               |
| VLLM              | 72.63    | 07:52                    | 3.1                               |

As shown in the table, using VLLM or LMDeploy for inference acceleration can significantly reduce inference time while maintaining high accuracy.
