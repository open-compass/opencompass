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
export LMDEPLOY_VERSION=0.6.0
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
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_a58960 import \
        gsm8k_datasets
    # and output the results in a chosen format
    from .summarizers.medium import summarizer

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# configure lmdeploy
from opencompass.models import TurboMindModelwithChatTemplate



# configure the model
models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr=f'internlm2-chat-7b-lmdeploy',
        # model path, which can be the address of a model repository on the Hugging Face Hub or a local path
        path='internlm/internlm2-chat-7b',
        # inference backend of LMDeploy. It can be either 'turbomind' or 'pytorch'.
        # If the model is not supported by 'turbomind', it will fallback to
        # 'pytorch'
        backend='turbomind',
        # For the detailed engine config and generation config, please refer to
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/messages.py
        engine_config=dict(tp=1),
        gen_config=dict(do_sample=False),
        # the max size of the context window
        max_seq_len=7168,
        # the max number of new tokens
        max_out_len=1024,
        # the max number of prompts that LMDeploy receives
        # in `generate` function
        batch_size=5000,
        run_cfg=dict(num_gpus=1),
    )
]
```

把上述配置放在文件中，比如 "configs/eval_internlm2_lmdeploy.py"。然后，在 OpenCompass 的项目目录下，执行如下命令可得到评测结果：

```shell
python run.py configs/eval_internlm2_lmdeploy.py -w outputs
```
