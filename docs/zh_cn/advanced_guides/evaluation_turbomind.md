# 使用 LMDeploy 加速评测

我们支持在评测大语言模型时，使用 [LMDeploy](https://github.com/InternLM/lmdeploy) 作为推理加速引擎。LMDeploy 是涵盖了 LLM 和 VLM 任务的全套轻量化、部署和服务解决方案，拥有卓越的推理性能。本教程将介绍如何使用 LMDeploy 加速对模型的评测。

## 环境配置

### 安装 OpenCompass

请根据 OpenCompass [安装指南](https://opencompass.readthedocs.io/en/latest/get_started/installation.html) 来安装算法库和准备数据集。

### 安装 LMDeploy

使用 pip 安装 LMDeploy (python 3.8+)：

```shell
pip install lmdeploy
```

LMDeploy 预编译包默认基于 CUDA 12 编译。如果需要在 CUDA 11+ 下安装 LMDeploy，请执行以下命令：

```shell
export LMDEPLOY_VERSION=0.4.2
export PYTHON_VERSION=310
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

## 评测

在评测一个模型时，需要准备一份评测配置，指明评测集、模型和推理参数等信息。

以 [internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) 模型为例，相关的配置信息如下：

```python
# configure the dataset
from mmengine.config import read_base


with read_base():
    # choose a list of datasets
    from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from .datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # and output the results in a chosen format
    from .summarizers.medium import summarizer

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# configure lmdeploy
from opencompass.models import LMDeploywithChatTemplate

# the engine name of lmdeploy, it can be one of ['turbomind', 'pytorch']
backend = 'turbomind'
# the config of the inference backend
# For the detailed config, please refer to
# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/messages.py
engine_config = dict(
    backend=backend,
    turbomind=dict(
        max_batch_size=128,
        tp=1,
        quant_policy='0',
        model_format='hf',
        enable_prefix_caching=False,
    ),
    pytorch=dict(
        max_batch_size=128,
        tp=1,
        enable_prefix_caching=False,
    )
)

# configure the model
models = [
    dict(
        type=LMDeploywithChatTemplate,
        abbr=f'internlm2-chat-7b-{backend}',
        # model path, which can be the address of a model repository on the Hugging Face Hub or a local path
        path='internlm/internlm2-chat-7b',
        engine_config=engine_config,
        gen_config=dict(
            # top_k=1 indicates greedy-search sampling
            top_k=1,
        ),
        # the max size of the context window
        max_seq_len=8000,
        # the max number of new tokens
        max_out_len=1024,
        # the max number of prompts passed to lmdeploy inference api
        batch_size=5000,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
    )
]
```

把上述配置放在文件中，比如 "configs/eval_internlm2_lmdeploy.py"。然后，在 OpenCompass 的项目目录下，执行如下命令可得到评测结果：

```shell
python run.py configs/eval_internlm2_lmdeploy.py -w outputs
```
