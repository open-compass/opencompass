# 评测LMDeploy模型

我们支持评测使用[LMDeploy](https://github.com/InternLM/lmdeploy)加速过的大语言模型。LMDeploy 由 MMDeploy 和 MMRazor 团队联合开发，是涵盖了 LLM 任务的全套轻量化、部署和服务解决方案。 **TurboMind** 是LMDeploy推出的高效推理引擎。OpenCompass对TurboMind进行了适配，本教程将介绍如何使用OpenCompass来对TurboMind加速后的模型进行评测。

# 环境配置

## 安装OpenCompass

请根据OpenCompass[安装指南](https://opencompass.readthedocs.io/en/latest/get_started.html) 来安装算法库和准备数据集。

## 安装LMDeploy

使用pip安装LMDeploy( python 3.8+)

```shell
pip install lmdeploy
```

# 评测

我们使用InternLM作为例子来介绍如何评测

## 第一步: 获取InternLM模型

```shell
# 1. Download InternLM model(or use the cached model's checkpoint)

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/internlm/internlm-chat-7b /path/to/internlm-chat-7b

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1

# 2. Convert InternLM model to turbomind's format, which will be in "./workspace" by default
python3 -m lmdeploy.serve.turbomind.deploy internlm-chat-7b /path/to/internlm-chat-7b

```

## 第二步: 验证转换后的模型

```shell
python -m lmdeploy.turbomind.chat ./workspace
```

## 第三步: 评测转换后的模型

在OpenCompass项目文件执行：

```shell
python run.py configs/eval_internlm_chat_7b_turbomind.py -w outputs/turbomind
```

当模型完成推理和指标计算后，我们便可获得模型的评测结果
