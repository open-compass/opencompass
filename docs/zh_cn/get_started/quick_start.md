# 快速开始

![image](https://github.com/open-compass/opencompass/assets/22607038/d063cae0-3297-4fd2-921a-366e0a24890b)

## 概览

在 OpenCompass 中评估一个模型通常包括以下几个阶段：**配置** -> **推理** -> **评估** -> **可视化**。

**配置**：这是整个工作流的起点。您需要配置整个评估过程，选择要评估的模型和数据集。此外，还可以选择评估策略、计算后端等，并定义显示结果的方式。

**推理与评估**：在这个阶段，OpenCompass 将会开始对模型和数据集进行并行推理和评估。**推理**阶段主要是让模型从数据集产生输出，而**评估**阶段则是衡量这些输出与标准答案的匹配程度。这两个过程会被拆分为多个同时运行的“任务”以提高效率，但请注意，如果计算资源有限，这种策略可能会使评测变得更慢。如果需要了解该问题及解决方案，可以参考 [FAQ: 效率](faq.md#效率)。

**可视化**：评估完成后，OpenCompass 将结果整理成易读的表格，并将其保存为 CSV 和 TXT 文件。你也可以激活飞书状态上报功能，此后可以在飞书客户端中及时获得评测状态报告。

接下来，我们将展示 OpenCompass 的基础用法，展示基座模型模型 [InternLM2-1.8B](https://huggingface.co/internlm/internlm2-1_8b) 和对话模型 [InternLM2-Chat-1.8B](https://huggingface.co/internlm/internlm2-chat-1_8b)、[Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) 在 [GSM8K](https://github.com/openai/grade-school-math) 和 [MATH](https://github.com/hendrycks/math) 下采样数据集上的评估。它们的配置文件可以在 [configs/eval_chat_demo.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_chat_demo.py) 和 [configs/eval_base_demo.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_base_demo.py) 中找到。

在运行此实验之前，请确保您已在本地安装了 OpenCompass。这个例子 (应该) 可以在一台 _GTX-1660-6G_ GPU 下成功运行。

对于参数更大的模型，如 Llama3-8B，请参考 [configs 目录](https://github.com/open-compass/opencompass/tree/main/configs) 中提供的其他示例。

## 配置评估任务

在 OpenCompass 中，每个评估任务由待评估的模型和数据集组成。评估的入口点是 `run.py`。用户可以通过命令行或配置文件选择要测试的模型和数据集。

对于对话模型

`````{tabs}
````{tab} 命令行（自定义 HF 模型）

对于 HuggingFace 模型，用户可以通过命令行直接设置模型参数，无需额外的配置文件。例如，对于 `internlm/internlm2-chat-1_8b` 模型，您可以使用以下命令进行评估：

```bash
python run.py \
    --datasets demo_gsm8k_chat_gen demo_math_chat_gen \
    --hf-type chat \
    --hf-path internlm/internlm2-chat-1_8b \
    --debug
```

请注意，通过这种方式，OpenCompass 一次只评估一个模型，而其他方式可以一次评估多个模型。

:::{dropdown} HF 模型完整参数列表
:animate: fade-in-slide-down
| 命令行参数 | 描述 | 样例数值 |
| --- | --- | --- |
| `--hf-type` | HuggingFace 模型类型，可选值为 `chat` 或 `base` | chat |
| `--hf-path` | HuggingFace 模型路径 | internlm/internlm2-chat-1_8b |
| `--model-kwargs` | 构建模型的参数 | device_map='auto' |
| `--tokenizer-path` | HuggingFace tokenizer 路径（如果与模型路径相同，可以省略） | internlm/internlm2-chat-1_8b |
| `--tokenizer-kwargs` | 构建 tokenizer 的参数 | padding_side='left' truncation='left' trust_remote_code=True |
| `--generation-kwargs` | 生成的参数 | do_sample=True top_k=50 top_p=0.95 |
| `--max-seq-len` | 模型可以接受的最大序列长度 | 2048 |
| `--max-out-len` | 生成的最大 token 数 | 100 |
| `--min-out-len` | 生成的最小 token 数 | 1 |
| `--batch-size` | 批量大小 | 64 |
| `--hf-num-gpus` | 运行一个模型实例所需的 GPU 数量 | 1 |
| `--stop-words` | 停用词列表 | '<\|im_end\|>' '<\|im_start\|>' |
| `--pad-token-id` | 填充 token 的 ID | 0 |
| `--peft-path` | (例如) LoRA 模型的路径 | internlm/internlm2-chat-1_8b |
| `--peft-kwargs` | (例如) 构建 LoRA 模型的参数 | trust_remote_code=True |
:::

:::{dropdown} 更复杂的命令样例
:animate: fade-in-slide-down

例如一个占用 2 卡进行测试的 Qwen1.5-14B-Chat, 开启数据采样，模型的命令如下：

```bash
python run.py --datasets demo_gsm8k_chat_gen demo_math_chat_gen \
    --hf-type chat \
    --hf-path Qwen/Qwen1.5-14B-Chat \
    --max-out-len 1024 \
    --min-out-len 1 \
    --hf-num-gpus 2 \
    --generation-kwargs do_sample=True temperature=0.6 \
    --stop-words '<|im_end|>' '<|im_start|>' \
    --debug
```
:::

````
````{tab} 命令行

用户可以使用 `--models` 和 `--datasets` 结合想测试的模型和数据集。

```bash
python run.py \
    --models hf_internlm2_chat_1_8b hf_qwen2_1_5b_instruct \
    --datasets demo_gsm8k_chat_gen demo_math_chat_gen \
    --debug
```

模型和数据集的配置文件预存于 `configs/models` 和 `configs/datasets` 中。用户可以使用 `tools/list_configs.py` 查看或过滤当前可用的模型和数据集配置。

```bash
# 列出所有配置
python tools/list_configs.py
# 列出与llama和mmlu相关的所有配置
python tools/list_configs.py llama mmlu
```

:::{dropdown} 关于 `list_configs`
:animate: fade-in-slide-down

运行 `python tools/list_configs.py llama mmlu` 将产生如下输出：

```text
+-----------------+-----------------------------------+
| Model           | Config Path                       |
|-----------------+-----------------------------------|
| hf_llama2_13b   | configs/models/hf_llama2_13b.py   |
| hf_llama2_70b   | configs/models/hf_llama2_70b.py   |
| ...             | ...                               |
+-----------------+-----------------------------------+
+-------------------+---------------------------------------------------+
| Dataset           | Config Path                                       |
|-------------------+---------------------------------------------------|
| cmmlu_gen         | configs/datasets/cmmlu/cmmlu_gen.py               |
| cmmlu_gen_ffe7c0  | configs/datasets/cmmlu/cmmlu_gen_ffe7c0.py        |
| ...               | ...                                               |
+-------------------+---------------------------------------------------+
```

用户可以使用第一列中的名称作为 `python run.py` 中 `--models` 和 `--datasets` 的输入参数。对于数据集，同一名称的不同后缀通常表示其提示或评估方法不同。
:::

:::{dropdown} 没有列出的模型？
:animate: fade-in-slide-down

如果您想评估其他模型，请查看 “命令行（自定义 HF 模型）”选项卡，了解无需配置文件自定义 HF 模型的方法，或 “配置文件”选项卡，了解准备模型配置的通用方法。

:::

````
````{tab} 配置文件

除了通过命令行配置实验外，OpenCompass 还允许用户在配置文件中编写实验的完整配置，并通过 `run.py` 直接运行它。配置文件是以 Python 格式组织的，并且必须包括 `datasets` 和 `models` 字段。

本次测试配置在 [configs/eval_chat_demo.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_chat_demo.py) 中。此配置通过 [继承机制](../user_guides/config.md#继承机制) 引入所需的数据集和模型配置，并以所需格式组合 `datasets` 和 `models` 字段。

```python
from mmengine.config import read_base

with read_base():
    from .datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
    from .datasets.demo.demo_math_chat_gen import math_datasets
    from .models.qwen.hf_qwen2_1_5b_instruct import models as hf_qwen2_1_5b_instruct_models
    from .models.hf_internlm.hf_internlm2_chat_1_8b import models as hf_internlm2_chat_1_8b_models

datasets = gsm8k_datasets + math_datasets
models = hf_qwen2_1_5b_instruct_models + hf_internlm2_chat_1_8b_models
```

运行任务时，我们只需将配置文件的路径传递给 `run.py`：

```bash
python run.py configs/eval_chat_demo.py --debug
```

:::{dropdown} 关于 `models`
:animate: fade-in-slide-down

OpenCompass 提供了一系列预定义的模型配置，位于 `configs/models` 下。以下是与 [InternLM2-Chat-1.8B](https://github.com/open-compass/opencompass/blob/main/configs/models/hf_internlm/hf_internlm2_chat_1_8b.py)（`configs/models/hf_internlm/hf_internlm2_chat_1_8b.py`）相关的配置片段：

```python
# 使用 `HuggingFacewithChatTemplate` 评估由 HuggingFace 的 `AutoModelForCausalLM` 支持的对话模型
from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2-chat-1.8b-hf',         # 模型的缩写
        path='internlm/internlm2-chat-1_8b',   # 模型的 HuggingFace 路径
        max_out_len=1024,                      # 生成的最大 token 数
        batch_size=8,                          # 批量大小
        run_cfg=dict(num_gpus=1),              # 该模型所需的 GPU 数量
    )
]
```

使用配置时，我们可以通过命令行参数 `--models` 指定相关文件，或使用继承机制将模型配置导入到配置文件中的 `models` 列表中。

```{seealso}
有关模型配置的更多信息，请参见 [准备模型](../user_guides/models.md)。
```
:::

:::{dropdown} 关于 `datasets`
:animate: fade-in-slide-down

与模型类似，数据集的配置文件也提供在 `configs/datasets` 下。用户可以在命令行中使用 `--datasets`，或通过继承在配置文件中导入相关配置

下面是来自 `configs/eval_chat_demo.py` 的与数据集相关的配置片段：

```python
from mmengine.config import read_base  # 使用 mmengine.read_base() 读取基本配置

with read_base():
    # 直接从预设的数据集配置中读取所需的数据集配置
    from .datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets  # 读取 GSM8K 配置，使用 4-shot，基于生成式进行评估
    from .datasets.demo.demo_math_chat_gen import math_datasets    # 读取 MATH 配置，使用 0-shot，基于生成式进行评估

datasets = gsm8k_datasets + math_datasets       # 最终的配置需要包含所需的评估数据集列表 'datasets'
```

数据集配置通常有两种类型：`ppl` 和 `gen`，分别指示使用的评估方法。其中 `ppl` 表示辨别性评估，`gen` 表示生成性评估。对话模型仅使用 `gen` 生成式评估。

此外，[configs/datasets/collections](https://github.com/open-compass/opencompass/blob/main/configs/datasets/collections) 收录了各种数据集集合，方便进行综合评估。OpenCompass 通常使用 [`chat_OC15.py`](https://github.com/open-compass/opencompass/blob/main/configs/dataset_collections/chat_OC15.py) 进行全面的模型测试。要复制结果，只需导入该文件，例如：

```bash
python run.py --models hf_internlm2_chat_1_8b --datasets chat_OC15 --debug
```

```{seealso}
您可以从 [配置数据集](../user_guides/datasets.md) 中找到更多信息。
```
:::

````
`````

对于基座模型

`````{tabs}
````{tab} 命令行（自定义 HF 模型）

对于 HuggingFace 模型，用户可以通过命令行直接设置模型参数，无需额外的配置文件。例如，对于 `internlm/internlm2-1_8b` 模型，您可以使用以下命令进行评估：

```bash
python run.py \
    --datasets demo_gsm8k_base_gen demo_math_base_gen \
    --hf-type base \
    --hf-path internlm/internlm2-1_8b \
    --debug
```

请注意，通过这种方式，OpenCompass 一次只评估一个模型，而其他方式可以一次评估多个模型。

:::{dropdown} 更复杂的命令样例
:animate: fade-in-slide-down

例如一个占用 2 卡进行测试的 Qwen1.5-14B, 开启数据采样，模型的命令如下：

```bash
python run.py --datasets demo_gsm8k_base_gen demo_math_base_gen \
    --hf-type chat \
    --hf-path Qwen/Qwen1.5-14B \
    --max-out-len 1024 \
    --min-out-len 1 \
    --hf-num-gpus 2 \
    --generation-kwargs do_sample=True temperature=0.6 \
    --debug
```
:::

````
````{tab} 命令行

用户可以使用 `--models` 和 `--datasets` 结合想测试的模型和数据集。

```bash
python run.py \
    --models hf_internlm2_1_8b hf_qwen2_1_5b \
    --datasets demo_gsm8k_base_gen demo_math_base_gen \
    --debug
```

````
````{tab} 配置文件

除了通过命令行配置实验外，OpenCompass 还允许用户在配置文件中编写实验的完整配置，并通过 `run.py` 直接运行它。配置文件是以 Python 格式组织的，并且必须包括 `datasets` 和 `models` 字段。

本次测试配置在 [configs/eval_base_demo.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_base_demo.py) 中。此配置通过 [继承机制](../user_guides/config.md#继承机制) 引入所需的数据集和模型配置，并以所需格式组合 `datasets` 和 `models` 字段。

```python
from mmengine.config import read_base

with read_base():
    from .datasets.demo.demo_gsm8k_base_gen import gsm8k_datasets
    from .datasets.demo.demo_math_base_gen import math_datasets
    from .models.qwen.hf_qwen2_1_5b import models as hf_qwen2_1_5b_models
    from .models.hf_internlm.hf_internlm2_1_8b import models as hf_internlm2_1_8b_models

datasets = gsm8k_datasets + math_datasets
models = hf_qwen2_1_5b_models + hf_internlm2_1_8b_models
```

运行任务时，我们只需将配置文件的路径传递给 `run.py`：

```bash
python run.py configs/eval_base_demo.py --debug
```

:::{dropdown} 关于 `models`
:animate: fade-in-slide-down

OpenCompass 提供了一系列预定义的模型配置，位于 `configs/models` 下。以下是与 [InternLM2-1.8B](https://github.com/open-compass/opencompass/blob/main/configs/models/hf_internlm/hf_internlm2_1_8b.py)（`configs/models/hf_internlm/hf_internlm2_1_8b.py`）相关的配置片段：

```python
# 使用 `HuggingFaceBaseModel` 评估由 HuggingFace 的 `AutoModelForCausalLM` 支持的基座模型
from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='internlm2-1.8b-hf',              # 模型的缩写
        path='internlm/internlm2-1_8b',        # 模型的 HuggingFace 路径
        max_out_len=1024,                      # 生成的最大 token 数
        batch_size=8,                          # 批量大小
        run_cfg=dict(num_gpus=1),              # 该模型所需的 GPU 数量
    )
]
```

使用配置时，我们可以通过命令行参数 `--models` 指定相关文件，或使用继承机制将模型配置导入到配置文件中的 `models` 列表中。

```{seealso}
有关模型配置的更多信息，请参见 [准备模型](../user_guides/models.md)。
```
:::

:::{dropdown} 关于 `datasets`
:animate: fade-in-slide-down

与模型类似，数据集的配置文件也提供在 `configs/datasets` 下。用户可以在命令行中使用 `--datasets`，或通过继承在配置文件中导入相关配置

下面是来自 `configs/eval_base_demo.py` 的与数据集相关的配置片段：

```python
from mmengine.config import read_base  # 使用 mmengine.read_base() 读取基本配置

with read_base():
    # 直接从预设的数据集配置中读取所需的数据集配置
    from .datasets.demo.demo_gsm8k_base_gen import gsm8k_datasets  # 读取 GSM8K 配置，使用 4-shot，基于生成式进行评估
    from .datasets.demo.demo_math_base_gen import math_datasets    # 读取 MATH 配置，使用 0-shot，基于生成式进行评估

datasets = gsm8k_datasets + math_datasets       # 最终的配置需要包含所需的评估数据集列表 'datasets'
```

数据集配置通常有两种类型：`ppl` 和 `gen`，分别指示使用的评估方法。其中 `ppl` 表示判别性评估，`gen` 表示生成性评估。基座模型对于 "选择题" 类型的数据集会使用 `ppl` 判别性评估，其他则会使用 `gen` 生成式评估。

```{seealso}
您可以从 [配置数据集](../user_guides/datasets.md) 中找到更多信息。
```
:::

````
`````

```{warning}
OpenCompass 通常假定运行环境网络是可用的。如果您遇到网络问题或希望在离线环境中运行 OpenCompass，请参阅 [FAQ - 网络 - Q1](./faq.md#网络) 寻求解决方案。
```

接下来的部分将使用基于配置的方法，评测对话模型，作为示例来解释其他特征。

## 启动评估

由于 OpenCompass 默认并行启动评估过程，我们可以在第一次运行时以 `--debug` 模式启动评估，并检查是否存在问题。包括在前述的所有文档中，我们都使用了 `--debug` 开关。在 `--debug` 模式下，任务将按顺序执行，并实时打印输出。

```bash
python run.py configs/eval_chat_demo.py -w outputs/demo --debug
```

对话默写 'internlm/internlm2-chat-1_8b' 和 'Qwen/Qwen2-1.5B-Instruct' 将在首次运行期间从 HuggingFace 自动下载。
如果一切正常，您应该看到屏幕上显示 “Starting inference process”，且进度条开始前进：

```bash
[2023-07-12 18:23:55,076] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...
```

然后，您可以按 `Ctrl+C` 中断程序，并以正常模式运行以下命令：

```bash
python run.py configs/eval_chat_demo.py -w outputs/demo
```

在正常模式下，评估任务将在后台并行执行，其输出将被重定向到输出目录 `outputs/demo/{TIMESTAMP}`。前端的进度条只指示已完成任务的数量，而不考虑其成功或失败。**任何后端任务失败都只会在终端触发警告消息。**

:::{dropdown} `run.py` 中的更多参数
:animate: fade-in-slide-down
以下是与评估相关的一些参数，可以帮助您根据环境配置更有效的推理任务：

- `-w outputs/demo`：保存评估日志和结果的工作目录。在这种情况下，实验结果将保存到 `outputs/demo/{TIMESTAMP}`。
- `-r {TIMESTAMP/latest}`：重用现有的推理结果，并跳过已完成的任务。如果后面跟随时间戳，将重用工作空间路径下该时间戳的结果；若给定 latest 或干脆不指定，将重用指定工作空间路径下的最新结果。
- `--mode all`：指定任务的特定阶段。
  - all：（默认）执行完整评估，包括推理和评估。
  - infer：在每个数据集上执行推理。
  - eval：根据推理结果进行评估。
  - viz：仅显示评估结果。
- `--max-num-workers 8`：并行任务的最大数量。在如 Slurm 之类的分布式环境中，此参数指定提交任务的最大数量。在本地环境中，它指定同时执行的任务的最大数量。请注意，实际的并行任务数量取决于可用的 GPU 资源，可能不等于这个数字。

如果您不是在本地机器上执行评估，而是使用 Slurm 集群，您可以指定以下参数：

- `--slurm`：在集群上使用 Slurm 提交任务。
- `--partition(-p) my_part`：Slurm 集群分区。
- `--retry 2`：失败任务的重试次数。

```{seealso}
入口还支持将任务提交到阿里巴巴深度学习中心（DLC），以及更多自定义评估策略。请参考 [评测任务发起](../user_guides/experimentation.md#评测任务发起) 了解详情。
```

:::

## 可视化评估结果

评估完成后，评估结果表格将打印如下：

```text
dataset     version    metric    mode      qwen2-1.5b-instruct-hf    internlm2-chat-1.8b-hf
----------  ---------  --------  ------  ------------------------  ------------------------
demo_gsm8k  1d7fe4     accuracy  gen                        56.25                     32.81
demo_math   393424     accuracy  gen                        18.75                     14.06
```

所有运行输出将定向到 `outputs/demo/` 目录，结构如下：

```text
outputs/default/
├── 20200220_120000
├── 20230220_183030     # 每个实验一个文件夹
│   ├── configs         # 用于记录的已转储的配置文件。如果在同一个实验文件夹中重新运行了不同的实验，可能会保留多个配置
│   ├── logs            # 推理和评估阶段的日志文件
│   │   ├── eval
│   │   └── infer
│   ├── predictions   # 每个任务的推理结果
│   ├── results       # 每个任务的评估结果
│   └── summary       # 单个实验的汇总评估结果
├── ...
```

打印评测结果的过程可被进一步定制化，用于输出一些数据集的平均分 (例如 MMLU, C-Eval 等)。

关于评测结果输出的更多介绍可阅读 [结果展示](../user_guides/summarizer.md)。

## 更多教程

想要更多了解 OpenCompass, 可以点击下列链接学习。

- [配置数据集](../user_guides/datasets.md)
- [准备模型](../user_guides/models.md)
- [任务运行和监控](../user_guides/experimentation.md)
- [如何调 Prompt](../prompt/overview.md)
- [结果展示](../user_guides/summarizer.md)
- [学习配置文件](../user_guides/config.md)
