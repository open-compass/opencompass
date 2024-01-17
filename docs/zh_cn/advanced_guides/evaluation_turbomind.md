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

OpenCompass 支持分别通过 turbomind python API 评测数据集。

下文以 InternLM-20B 模型为例，介绍如何评测。

在 OpenCompass 的项目目录下，执行如下命令可得到评测结果：

```shell
python run.py configs/eval_internlm_turbomind.py -w outputs/turbomind/internlm-20b
```

**注：**

- 如果评测 InternLM Chat 模型，请使用配置文件 `eval_internlm_chat_turbomind.py`
- 如果评测 InternLM 7B 模型，请修改 `eval_internlm_turbomind.py` 或者 `eval_internlm_chat_turbomind.py`。将`models`字段配置为`models = [internlm_7b]` 。
