# 安装

## 基础安装

1. 使用Conda准备 OpenCompass 运行环境：

   ```bash
   conda create --name opencompass python=3.10 -y
   # conda create --name opencompass_lmdeploy python=3.10 -y

   conda activate opencompass
   ```

   如果你希望自定义 PyTorch 版本，请参考 [官方文档](https://pytorch.org/get-started/locally/) 准备 PyTorch 环境。需要注意的是，OpenCompass 要求 `pytorch>=1.13`。

2. 安装 OpenCompass：

   - pip安装

   ```bash
   # 支持绝大多数数据集及模型
   pip install -U opencompass

   # 完整安装（支持更多数据集）
   # pip install "opencompass[full]"

   # API 测试（例如 OpenAI、Qwen）
   # pip install "opencompass[api]"
   ```

   - 如果希望使用 OpenCompass 的最新功能，也可以从源代码构建它：

   ```bash
   git clone https://github.com/open-compass/opencompass opencompass
   cd opencompass
   pip install -e .
   ```

## 其他安装

### 推理后端

```bash
# 模型推理后端，由于这些推理后端通常存在依赖冲突，建议使用不同的虚拟环境来管理它们。
pip install "opencompass[lmdeploy]"
# pip install "opencompass[vllm]"
```

- LMDeploy

可以通过下列命令判断推理后端是否安装成功，更多信息请参考 [官方文档](https://lmdeploy.readthedocs.io/zh-cn/latest/get_started.html)

```bash
lmdeploy chat internlm/internlm2_5-1_8b-chat --backend turbomind
```

- vLLM
  可以通过下列命令判断推理后端是否安装成功，更多信息请参考 [官方文档](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)

```bash
vllm serve facebook/opt-125m
```

### API

Opencompass支持不同的商业模型API调用，你可以通过pip方式安装，或者参考 [API](https://github.com/open-compass/opencompass/blob/main/requirements/api.txt) 安装对应的API模型依赖

```bash
pip install opencompass[api]

# pip install openai # GPT-3.5-Turbo / GPT-4-Turbo / GPT-4 / GPT-4o (API)
# pip install anthropic # Claude (API)
# pip install dashscope #  通义千问 (API)
# pip install volcengine-python-sdk # 字节豆包 (API)
# ...
```

### 数据集

基础安装可以支持绝大部分基础数据集，针对某些数据集（i.e. Alpaca-eval, Longbench etc.），需要安装额外的依赖。
你可以通过pip方式安装，或者参考 [额外依赖](https://github.com/open-compass/opencompass/blob/main/requirements/extra.txt) 安装对应的依赖

```bash
pip install opencompass[full]
```

针对 HumanEvalX / HumanEval+ / MBPP+ 需要手动clone git仓库进行安装

```bash
git clone --recurse-submodules git@github.com:open-compass/human-eval.git
cd human-eval
pip install -e .
pip install -e evalplus
```

部分智能体评测需要安装大量依赖且可能会与已有运行环境冲突，我们建议创建不同的conda环境来管理

```bash
# T-Eval
pip install lagent==0.1.2
# CIBench
pip install -r requirements/agent.txt
```

## 数据集准备

OpenCompass 支持的数据集主要包括三个部分：

1. Huggingface 数据集： [Huggingface Dataset](https://huggingface.co/datasets) 提供了大量的数据集，这部分数据集运行时会**自动下载**。

2. ModelScope 数据集：[ModelScope OpenCompass Dataset](https://modelscope.cn/organization/opencompass) 支持从 ModelScope 自动下载数据集。

   要启用此功能，请设置环境变量：`export DATASET_SOURCE=ModelScope`，可用的数据集包括（来源于 OpenCompassData-core.zip）：

   ```plain
   humaneval, triviaqa, commonsenseqa, tydiqa, strategyqa, cmmlu, lambada, piqa, ceval, math, LCSTS, Xsum, winogrande, openbookqa, AGIEval, gsm8k, nq, race, siqa, mbpp, mmlu, hellaswag, ARC, BBH, xstory_cloze, summedits, GAOKAO-BENCH, OCNLI, cmnli
   ```

3. 自建以及第三方数据集：OpenCompass 还提供了一些第三方数据集及自建**中文**数据集。运行以下命令**手动下载解压**。

在 OpenCompass 项目根目录下运行下面命令，将数据集准备至 `${OpenCompass}/data` 目录下：

```bash
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

如果需要使用 OpenCompass 提供的更加完整的数据集 (~500M)，可以使用下述命令进行下载和解压：

```bash
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
unzip OpenCompassData-complete-20240207.zip
cd ./data
find . -name "*.zip" -exec unzip "{}" \;
```

两个 `.zip` 中所含数据集列表如[此处](https://github.com/open-compass/opencompass/releases/tag/0.2.2.rc1)所示。

OpenCompass 已经支持了大多数常用于性能比较的数据集，具体支持的数据集列表请直接在 `configs/datasets` 下进行查找。

接下来，你可以阅读[快速上手](./quick_start.md)了解 OpenCompass 的基本用法。
