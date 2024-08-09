# 其他安装说明

欢迎使用浏览器自带的搜索功能。

## 推理后端

OpenCompass 开发者所使用的 CUDA 版本为 12.1。

由于不同的推理后端会有依赖冲突，我们强烈建议使用conda管理包依赖环境

- LMDeploy

```bash
pip install -U opencompass[lmdeploy]
```

安装完成后，可运行以下命令确保安装成功

```bash
lmdeploy serve api_server internlm/internlm2_5-7b-chat
```

具体文档可见 [LMDeploy 官方文档](https://lmdeploy.readthedocs.io/zh-cn/latest/get_started.html#id3)

- vLLM

```bash
pip install -U opencompass[vllm]
```

安装完成后，可运行以下命令确保安装成功

```bash
vllm serve facebook/opt-125m
```

具体文档可见 [vLLM官方文档](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)

## 模型

- 开源模型

OpenCompass的基础依赖可以提供一个兼容大部分开源模型的环境，但是OpenCompass无法保证提供一个可以应对所有模型，针对某些有特殊依赖 (i.e. FlashAttention, XFormers) 的模型，用户可依据相关文档安装。

- 闭源API模型

```bash
pip install -r requirements/api.txt
```

## 用户也可根据需要，安装指定模型API

```bash
pip install openai # GPT-3.5-Turbo / GPT-4-Turbo / GPT-4 / GPT-4o (API)
pip install anthropic # Claude (API)
pip install dashscope #  通义千问 (API)
pip install volcengine-python-sdk # 字节豆包 (API)
...
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
