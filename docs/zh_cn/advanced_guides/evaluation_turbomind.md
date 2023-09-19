# 评测 LMDeploy 模型

我们支持评测使用 [LMDeploy](https://github.com/InternLM/lmdeploy) 加速过的大语言模型。LMDeploy 由 MMDeploy 和 MMRazor 团队联合开发，是涵盖了 LLM 任务的全套轻量化、部署和服务解决方案。 **TurboMind** 是 LMDeploy 推出的高效推理引擎。OpenCompass 对 TurboMind 进行了适配，本教程将介绍如何使用 OpenCompass 来对 TurboMind 加速后的模型进行评测。

## 环境配置

### 安装 OpenCompass

请根据 OpenCompass [安装指南](https://opencompass.readthedocs.io/en/latest/get_started.html) 来安装算法库和准备数据集。

### 安装 LMDeploy

使用 pip 安装 LMDeploy (python 3.8+)：

```shell
pip install lmdeploy
```

## 评测

我们使用 InternLM 作为例子来介绍如何评测。

### 第一步: 获取 InternLM 模型

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

### 第二步: 启动 TurboMind 的 Triton Inference Server

```shell
bash ./workspace/service_docker_up.sh
```

**注：** turbomind 的实现中，推理是“永驻”的。销毁操作会导致意想不到的问题发生。因此，我们暂时使用服务接口对接模型评测，待 turbomind 支持“销毁”之后，再提供 python API对接方式。

### 第三步: 评测转换后的模型

在 OpenCompass 项目目录执行：

```shell
python run.py configs/eval_internlm_chat_7b_turbomind.py -w outputs/turbomind
```

当模型完成推理和指标计算后，我们便可获得模型的评测结果。

**注：** `eval_internlm_chat_7b_turbomind.py` 中，配置的 triton inference server(TIS) 地址是 `tis_addr='0.0.0.0:63337'`。如果不在同一台机器上执行`run.py`，那么请把配置中的`tis_addr`修改为server所在机器的ip地址。
