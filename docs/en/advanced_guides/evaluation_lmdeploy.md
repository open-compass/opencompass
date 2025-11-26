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

Taking [Qwen3-0.6B]([https://huggingface.co/internlm/internlm2-chat-7b](https://huggingface.co/Qwen/Qwen3-0.6B)) as an example, the evaluation config is as follows:

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

from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen_3_0.6b_thinking-turbomind',
        path='Qwen/Qwen3-0.6B',
        engine_config=dict(session_len=32768, max_batch_size=16, tp=1),
        gen_config=dict(
            top_k=20, temperature=0.6, top_p=0.95, do_sample=True, enable_thinking=True
        ),
        max_seq_len=32768,
        max_out_len=32000,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    ),
]
```

Place the aforementioned configuration in a file, such as "configs/models/qwen3/lmdeploy_qwen3_0_6b.py". Then, in the home folder of OpenCompass, start evaluation by the following command:

```shell
python run.py configs/models/qwen3/lmdeploy_qwen3_0_6b.py -w outputs
```

You are expected to get the evaluation results after the inference and evaluation.
