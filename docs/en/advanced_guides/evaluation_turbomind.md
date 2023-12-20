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

OpenCompass integrates both turbomind's python API and gRPC API for evaluation. And the former is highly recommended.

We take the InternLM-20B as example. Please download it from huggingface and convert it to turbomind's model format:

```shell
# 1. Download InternLM model(or use the cached model's checkpoint)

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/internlm/internlm-20b /path/to/internlm-20b

# 2. Convert InternLM model to turbomind's format, and save it in the home folder of opencompass
lmdeploy convert internlm /path/to/internlm-20b \
    --dst-path {/home/folder/of/opencompass}/turbomind
```

**Note**:

If evaluating the InternLM Chat model, make sure to pass `internlm-chat` as the model name instead of `internlm` when converting the model format. The specific command is:

```shell
lmdeploy convert internlm-chat /path/to/internlm-20b-chat \
    --dst-path {/home/folder/of/opencompass}/turbomind
```

### Evaluation with Turbomind Python API (recommended)

In the home folder of OpenCompass, start evaluation by the following command:

```shell
python run.py configs/eval_internlm_turbomind.py -w outputs/turbomind/internlm-20b
```

You are expected to get the evaluation results after the inference and evaluation.

**Note**:

- If you evaluate the InternLM Chat model, please use configuration file `eval_internlm_chat_turbomind.py`
- If you evaluate the InternLM 7B model, please modify `eval_internlm_turbomind.py` or `eval_internlm_chat_turbomind.py` by changing to the setting `models = [internlm_7b]` in the last line.
- If you want to evaluate other chat models like Llama2, QWen-7B, Baichuan2-7B, you could change to the setting of `models` in `eval_internlm_chat_turbomind.py`.

### Evaluation with Turbomind gPRC API (optional)

In the home folder of OpenCompass, launch the Triton Inference Server:

```shell
bash turbomind/service_docker_up.sh
```

And start evaluation by the following command:

```shell
python run.py configs/eval_internlm_turbomind_tis.py -w outputs/turbomind-tis/internlm-20b
```

\*\*Note: \*\*

- If the InternLM Chat model is requested to be evaluated, please use config file `eval_internlm_chat_turbomind_tis.py`
- In `eval_internlm_turbomind_tis.py`, the configured Triton Inference Server (TIS) address is `tis_addr='0.0.0.0:33337'`. Please modify `tis_addr` to the IP address of the machine where the server is launched.
- If evaluating the InternLM 7B model, please modify the `models` configuration in `eval_internlm_xxx_turbomind_tis.py`.
