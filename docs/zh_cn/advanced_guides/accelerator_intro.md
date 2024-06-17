# 使用 VLLM 或 LMDeploy 来一键式加速评测推理

## 背景

在 OpenCompass 评测过程中，默认使用 Huggingface 的 transformers 库进行推理，这是一个非常通用的方案，但在某些情况下，我们可能需要更高效的推理方法来加速这一过程，比如借助 VLLM 或 LMDeploy。

- [LMDeploy](https://github.com/InternLM/lmdeploy) 是一个用于压缩、部署和服务大型语言模型（LLM）的工具包，由 [MMRazor](https://github.com/open-mmlab/mmrazor) 和 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 团队开发。
- [vLLM](https://github.com/vllm-project/vllm) 是一个快速且易于使用的 LLM 推理和服务库，具有先进的服务吞吐量、高效的 PagedAttention 内存管理、连续批处理请求、CUDA/HIP 图的快速模型执行、量化技术（如 GPTQ、AWQ、SqueezeLLM、FP8 KV Cache）以及优化的 CUDA 内核。

## 加速前准备

首先，请检查您要评测的模型是否支持使用 VLLM 或 LMDeploy 进行推理加速。其次，请确保您已经安装了 VLLM 或 LMDeploy，具体安装方法请参考它们的官方文档，下面是参考的安装方法：

### LMDeploy 安装方法

使用 pip (Python 3.8+) 或从 [源码](https://github.com/InternLM/lmdeploy/blob/main/docs/en/build.md) 安装 LMDeploy：

```bash
pip install lmdeploy
```

### VLLM 安装方法

使用 pip 或从 [源码](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source) 安装 vLLM：

```bash
pip install vllm
```

## 评测时使用 VLLM 或 LMDeploy

OpenCompass 提供了一键式的评测加速，可以在评测过程中自动将 Huggingface 的 transformers 模型转化为 VLLM 或 LMDeploy 的模型，以便在评测过程中使用。以下是使用默认 Huggingface 版本的 Internlm2-chat-7b 模型评测 GSM8k 数据集的脚本：

### OpenCompass 主库

```python
# eval_gsm8k.py
from mmengine.config import read_base

with read_base():
    # 选择一个数据集列表
    from .datasets.gsm8k.gsm8k_0shot_gen_a58960 import gsm8k_datasets as datasets
    # 选择一个感兴趣的模型
    from ..models.hf_llama.hf_llama3_8b_instruct import models
```

其中 `hf_llama3_8b_instruct` 为原版 Huggingface 模型配置，内容如下：

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

默认 Huggingface 版本的 Llama3-8b-instruct 模型评测 GSM8k 数据集的方式如下：

```bash
python run.py config/eval_gsm8k.py
```

如果需要使用 VLLM 或 LMDeploy 进行加速评测，可以使用下面的脚本：

```bash
python run.py config/eval_gsm8k.py -a vllm
```

或

```bash
python run.py config/eval_gsm8k.py -a lmdeploy
```

## 加速效果及性能对比

下面是使用 VLLM 或 LMDeploy 在单卡 A800 上对 GSM8k 数据集进行加速评测的效果及性能对比表：

| 推理后端    | 精度（Accuracy） | 推理时间（分钟：秒） | 加速比（相对于 Huggingface） |
| ----------- | ---------------- | -------------------- | ---------------------------- |
| Huggingface | 74.22            | 24:26                | 1.0                          |
| LMDeploy    | 73.69            | 11:15                | 2.2                          |
| VLLM        | 72.63            | 07:52                | 3.1                          |
