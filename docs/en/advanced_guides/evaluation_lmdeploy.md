# Evaluation with LMDeploy

We now support evaluation of models accelerated by the [LMDeploy](https://github.com/InternLM/lmdeploy). LMDeploy is a toolkit designed for compressing, deploying, and serving LLM. It has a remarkable inference performance. We now illustrate how to evaluate a model with the support of LMDeploy in OpenCompass.

## Setup

### Install OpenCompass

Please follow the [instructions](https://opencompass.readthedocs.io/en/latest/get_started/installation.html) to install the OpenCompass and prepare the evaluation datasets.

### Install LMDeploy

Install lmdeploy via pip (python 3.8+)

```shell
pip install lmdeploy
```

The default prebuilt package is compiled on CUDA 12. However, if CUDA 11+ is required, you can install lmdeploy by:

```shell
export LMDEPLOY_VERSION=0.6.0
export PYTHON_VERSION=310
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

## Evaluation

When evaluating a model, it is necessary to prepare an evaluation configuration that specifies information such as the evaluation dataset, the model, and inference parameters.

Taking [internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) as an example, the evaluation config is as follows:

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

Place the aforementioned configuration in a file, such as "configs/eval_internlm2_lmdeploy.py". Then, in the home folder of OpenCompass, start evaluation by the following command:

```shell
python run.py configs/eval_internlm2_lmdeploy.py -w outputs
```

You are expected to get the evaluation results after the inference and evaluation.
