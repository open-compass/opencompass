# 使用 vLLM、LMDeploy、SGLang 或 OpenAI 来一键式加速评测推理

## 背景

在 OpenCompass 评测过程中，默认使用 Huggingface 的 transformers 库进行推理，这是一个非常通用的方案，但在某些情况下，我们可能需要更高效的推理方法来加速这一过程，比如借助 VLLM、LMDeploy、SGLang 或 OpenAI 兼容的 API。

- [LMDeploy](https://github.com/InternLM/lmdeploy) 是一个用于压缩、部署和服务大型语言模型（LLM）的工具包，由 [MMRazor](https://github.com/open-mmlab/mmrazor) 和 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 团队开发。
- [vLLM](https://github.com/vllm-project/vllm) 是一个快速且易于使用的 LLM 推理和服务库，具有先进的服务吞吐量、高效的 PagedAttention 内存管理、连续批处理请求、CUDA/HIP 图的快速模型执行、量化技术（如 GPTQ、AWQ、SqueezeLLM、FP8 KV Cache）以及优化的 CUDA 内核。
- [SGLang](https://github.com/sgl-project/sglang) 是一个为大型语言模型（LLM）设计的结构化生成语言。它使您与模型的交互更快、更可控。
- **OpenAI 兼容 API** 允许您使用任何 OpenAI 兼容的端点进行模型推理，包括官方 OpenAI 模型或自托管的具有 OpenAI 兼容 API 接口的模型。

## 加速前准备

首先，请检查您要评测的模型是否支持使用 vLLM、LMDeploy、SGLang 或 OpenAI 兼容 API 进行推理加速。其次，请确保您已经安装了所需的后端，具体安装方法请参考它们的官方文档，下面是参考的安装方法：

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

### SGLang 安装方法

使用 pip 或从 [源码](https://github.com/sgl-project/sglang) 安装 SGLang：

```bash
pip install sglang
```

### OpenAI API 设置

对于 OpenAI 兼容的 API，您只需要安装 openai 包：

```bash
pip install openai
```

## 评测时使用 VLLM、LMDeploy、SGLang 或 OpenAI

### 方法1：使用命令行参数来变更推理后端

OpenCompass 提供了一键式的评测加速，可以在评测过程中自动将 Huggingface 的 transformers 模型转化为 VLLM、LMDeploy、SGLang 或 OpenAI 的模型，以便在评测过程中使用。以下是使用默认 Huggingface 版本的 llama3-8b-instruct 模型评测 GSM8k 数据集的样例代码：

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

如果需要使用 vLLM、LMDeploy、SGLang 或 OpenAI 进行加速评测，可以使用下面的脚本：

**使用 vLLM：**
```bash
python run.py config/eval_gsm8k.py -a vllm
```

**使用 LMDeploy：**
```bash
python run.py config/eval_gsm8k.py -a lmdeploy
```

**使用 SGLang：**
```bash
python run.py config/eval_gsm8k.py -a sglang
```

**使用 OpenAI API：**
```bash
python run.py config/eval_gsm8k.py -a openai
```

注意：对于 OpenAI 后端，您可能需要在模型配置中配置额外的参数，如 `openai_api_base` 和 `api_key`。

### 方法2：通过部署推理加速服务API来加速评测

OpenCompass 还支持通过部署 vLLM、LMDeploy 或 SGLang 的推理加速服务 API 来加速评测，参考步骤如下:

1. 安装openai包：

```bash
pip install openai
```

2. 部署 vLLM、LMDeploy 或 SGLang 的推理加速服务 API，具体部署方法请参考它们的官方文档，下面以 LMDeploy 为例：

```bash
lmdeploy serve api_server meta-llama/Meta-Llama-3-8B-Instruct --model-name Meta-Llama-3-8B-Instruct --server-port 23333
```

api_server 启动时的参数可以通过命令行`lmdeploy serve api_server -h`查看。 比如，--tp 设置张量并行，--session-len 设置推理的最大上下文窗口长度，--cache-max-entry-count 调整 k/v cache 的内存使用比例等等。

3. 服务部署成功后，修改评测脚本，将模型配置中的路径改为部署的服务地址，如下：

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
        openai_api_base='http://0.0.0.0:23333/v1', # 服务地址
        path='Meta-Llama-3-8B-Instruct ', # 请求服务时的 model name
        tokenizer_path='meta-llama/Meta-Llama-3.1-8B-Instruct', # 请求服务时的 tokenizer name 或 path, 为None时使用默认tokenizer gpt-4
        rpm_verbose=True, # 是否打印请求速率
        meta_template=api_meta_template, # 服务请求模板
        query_per_second=1, # 服务请求速率
        max_out_len=1024, # 最大输出长度
        max_seq_len=4096, # 最大输入长度
        temperature=0.01, # 生成温度
        batch_size=8, # 批处理大小
        retry=3, # 重试次数
    )
]
```

## 加速效果及性能对比

下面是使用 VLLM 或 LMDeploy 在单卡 A800 上 Llama-3-8B-Instruct 模型对 GSM8k 数据集进行加速评测的效果及性能对比表：

| 推理后端    | 精度（Accuracy） | 推理时间（分钟：秒） | 加速比（相对于 Huggingface） |
| ----------- | ---------------- | -------------------- | ---------------------------- |
| Huggingface | 74.22            | 24:26                | 1.0                          |
| LMDeploy    | 73.69            | 11:15                | 2.2                          |
| VLLM        | 72.63            | 07:52                | 3.1                          |
