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

OpenCompass 支持分别通过 turbomind python API 和 gRPC API 评测数据集。我们强烈推荐使用前者进行评测。

下文以 InternLM-20B 模型为例，介绍如何评测。首先，从 huggingface 上下载 InternLM 模型，并转换为 turbomind 模型格式：

```shell
# 1. Download InternLM model(or use the cached model's checkpoint)

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/internlm/internlm-20b /path/to/internlm-20b

# 2. Convert InternLM model to turbomind's format, and save it in the home folder of opencompass
lmdeploy convert internlm /path/to/internlm-20b \
    --dst-path {/home/folder/of/opencompass}/turbomind
```

注意：如果评测 InternLM Chat 模型，那么在转换模型格式的时候，模型名字要填写 `internlm-chat`。具体命令是：

```shell
lmdeploy convert internlm-chat /path/to/internlm-20b-chat \
    --dst-path {/home/folder/of/opencompass}/turbomind
```

### 通过 TurboMind Python API 评测（推荐）

在 OpenCompass 的项目目录下，执行如下命令可得到评测结果：

```shell
python run.py configs/eval_internlm_turbomind.py -w outputs/turbomind/internlm-20b
```

**注：**

- 如果评测 InternLM Chat 模型，请使用配置文件 `eval_internlm_chat_turbomind.py`
- 如果评测 InternLM 7B 模型，请修改 `eval_internlm_turbomind.py` 或者 `eval_internlm_chat_turbomind.py`。将`models`字段配置为`models = [internlm_7b]` 。
- 如果评测其他模型如 Llama2, QWen-7B, Baichuan2-7B, 请修改`eval_internlm_chat_turbomind.py`中`models`字段 。

### 通过 TurboMind gPRC API 评测（可选）

在 OpenCompass 的项目目录下，启动 triton inference server：

```shell
bash turbomind/service_docker_up.sh
```

然后，执行如下命令进行评测：

```shell
python run.py configs/eval_internlm_turbomind_tis.py -w outputs/turbomind-tis/internlm-20b
``

**注：**

- 如果评测 InternLM Chat 模型，请使用配置文件 `eval_internlm_chat_turbomind_tis.py`
- 在配置文件中，triton inference server(TIS) 地址是 `tis_addr='0.0.0.0:33337'`。请把配置中的`tis_addr`修改为server所在机器的ip地址。
- 如果评测 InternLM 7B 模型，请修改 `eval_internlm_xxx_turbomind_tis.py`中`models`字段。
```
