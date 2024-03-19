# Evaluation with LMDeploy

We now support evaluation of models accelerated by the [LMDeploy](https://github.com/InternLM/lmdeploy). LMDeploy is a toolkit designed for compressing, deploying, and serving LLM. **TurboMind** is an efficient inference engine proposed by LMDeploy. OpenCompass is compatible with TurboMind. We now illustrate how to evaluate a model with the support of TurboMind in OpenCompass.

## Setup

### Install OpenCompass

Please follow the [instructions](https://opencompass.readthedocs.io/en/latest/get_started/installation.html) to install the OpenCompass and prepare the evaluation datasets.

### Install LMDeploy

Install lmdeploy via pip (python 3.8+)

```shell
pip install lmdeploy
```

## Evaluation

OpenCompass integrates turbomind's python API for evaluation.

We take the InternLM-20B as example. Firstly, we prepare the evaluation config `configs/eval_internlm_turbomind.py`:

```python
from mmengine.config import read_base
from opencompass.models.turbomind import TurboMindModel


with read_base():
    # choose a list of datasets
    from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from .datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    from .datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # and output the results in a chosen format
    from .summarizers.medium import summarizer

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# config for internlm-20b model
internlm_20b = dict(
        type=TurboMindModel,
        abbr='internlm-20b-turbomind',
        path="internlm/internlm-20b",  # this path should be same as in huggingface
        engine_config=dict(session_len=2048,
                           max_batch_size=8,
                           rope_scaling_factor=1.0),
        gen_config=dict(top_k=1, top_p=0.8,
                        temperature=1.0,
                        max_new_tokens=100),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        concurrency=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<eoa>'
    )

models = [internlm_20b]
```

Then, in the home folder of OpenCompass, start evaluation by the following command:

```shell
python run.py configs/eval_internlm_turbomind.py -w outputs/turbomind/internlm-20b
```

You are expected to get the evaluation results after the inference and evaluation.

**Note**:

- If you want to pass more arguments for `engine_config`和`gen_config` in the evaluation config file, please refer to [TurbomindEngineConfig](https://lmdeploy.readthedocs.io/en/latest/inference/pipeline.html#turbomindengineconfig)
  and [EngineGenerationConfig](https://lmdeploy.readthedocs.io/en/latest/inference/pipeline.html#generationconfig)
- If you evaluate the InternLM Chat model, please use configuration file `eval_internlm_chat_turbomind.py`
- If you evaluate the InternLM 7B model, please modify `eval_internlm_turbomind.py` or `eval_internlm_chat_turbomind.py` by changing to the setting `models = [internlm_7b]` in the last line.
