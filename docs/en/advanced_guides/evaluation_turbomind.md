# Evaluation with LMDeploy

We now support evaluation of models accelerated by the [LMDeploy](https://github.com/InternLM/lmdeploy). LMDeploy is a toolkit designed for compressing, deploying, and serving LLM. **TurboMind** is an efficient inference engine proposed by LMDeploy. OpenCompass is compatible with TurboMind. We now illustrate how to evaluate a model with the support of TurboMind in OpenCompass.

## Setup

### Install OpenCompass

Please follow the [instructions](https://opencompass.readthedocs.io/en/latest/get_started.html) to install the OpenCompass and prepare the evaluation datasets.

### Install LMDeploy

Install lmdeploy via pip (python 3.8+)

```shell
pip install lmdeploy
```

## Evaluation

OpenCompass integrates turbomind's python API for evaluation.

We take the InternLM-20B as example and prepare the evaluation config `configs/eval_internlm_turbomind.py`.

In the home folder of OpenCompass, start evaluation by the following command:

```shell
python run.py configs/eval_internlm_turbomind.py -w outputs/turbomind/internlm-20b
```

You are expected to get the evaluation results after the inference and evaluation.

**Note**:

- If you evaluate the InternLM Chat model, please use configuration file `eval_internlm_chat_turbomind.py`
- If you evaluate the InternLM 7B model, please modify `eval_internlm_turbomind.py` or `eval_internlm_chat_turbomind.py` by changing to the setting `models = [internlm_7b]` in the last line.
