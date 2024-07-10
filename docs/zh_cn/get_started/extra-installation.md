# 其他安装说明

欢迎使用浏览器自带的搜索功能。

## 推理后端

- LMDeploy

```bash
pip install lmdeploy
```

- VLLM

```bash
pip install vllm
```

OpenCompass 开发者所使用的 CUDA 版本为 11.8，一个能够应对 2024.07 之前绝大部分模型的依赖版本如下：

```bash
export VLLM_VERSION=0.4.3
export LMDEPLOY_VERSION=0.4.1
export FLASH_ATTN_VERSION=2.5.7
export XFORMERS_VERSION=0.0.25.post1
export TORCH_VERSION=2.2.2
export TORCHVISION_VERSION=0.17.2
export TORCHAUDIO_VERSION=2.2.2
export TRITON_VERSION=2.1.0
export PYTHON_VERSION=310


pip3 install "https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl" --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl" --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu118torch2.2cxx11abiFALSE-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl" --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install xformers==${XFORMERS_VERSION} --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu118
pip3 install triton==${TRITON_VERSION} --extra-index-url https://download.pytorch.org/whl/cu118
```

请注意，在安装过程中，后一条 `pip install` 命令可能会覆盖前一条命令中部分依赖的版本。并且在最终安装完成后，可能有的软件依赖会不满足，但是对 lmdeploy / vllm / xformers 等有需求的模型都是可以跑起来的。很神秘。

## 模型

- LLAMA (参数，原生, 非 HF 格式)

```bash
   git clone https://github.com/facebookresearch/llama.git
   cd llama
   pip install -r requirements.txt
   pip install -e .
```

- Vicuna (参数)

```bash
pip install "fschat[model_worker,webui]
```

- Baichuan / Baichuan2 (参数)

```bash
pip install "transformers<=4.33.3"
```

- ChatGLM-3 / GLM-4 (参数)

```bash
pip install "transformers<=4.41.2"
```

- GPT-3.5-Turbo / GPT-4-Turbo / GPT-4 / GPT-4o (API)

```bash
pip install openai
```

- Claude (API)

```bash
pip install anthropic
```

- 字节豆包 (API)

```bash
pip install volcengine-python-sdk
```

- 腾讯混元 (API)

```bash
pip install tencentcloud-sdk-python
```

- 讯飞星火 (API)

```bash
pip install spark_ai_python "sseclient-py==1.7.2"  websocket-client
```

- 智谱 (API)

```bash
pip install zhipuai
```

- 通义千问 (API)

```bash
pip install dashscope
```

## 数据集

- HumanEval

```bash
git clone git@github.com:open-compass/human-eval.git
cd human-eval && pip install -e .
```

该代码库 fork 自 https://github.com/openai/human-eval.git，并且已经注释了 `human_eval/execution.py` **第48-57行** 的提示。该提示告知了直接运行 LLM 生成的代码会有风险。

- HumanEvalX / HumanEval+ / MBPP+

```bash
git clone --recurse-submodules git@github.com:open-compass/human-eval.git
cd human-eval
pip install -e .
pip install -e evalplus
```

- AlpacaEval

```bash
pip install alpaca-eval==0.6 scikit-learn==1.5
```

- CIBench

```bash
pip install -r requirements/agent.txt
```

- T-Eval

```bash
pip install lagent==0.1.2
```

- APPS / TACO

```bash
pip install pyext
```

- IFEval

```bash
pip install langdetect
```

- NPHardEval

```bash
pip install networkx
```

- LawBench

```bash
pip install cn2an
```
