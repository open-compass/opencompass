# 快速开始

![image](https://github.com/open-compass/opencompass/assets/22607038/d063cae0-3297-4fd2-921a-366e0a24890b)

## 概览

在 OpenCompass 中评估一个模型通常包括以下几个阶段：**配置** -> **推理** -> **评估** -> **可视化**。

**配置**：这是整个工作流的起点。您需要配置整个评估过程，选择要评估的模型和数据集。此外，还可以选择评估策略、计算后端等，并定义显示结果的方式。

**推理与评估**：在这个阶段，OpenCompass 将会开始对模型和数据集进行并行推理和评估。**推理**阶段主要是让模型从数据集产生输出，而**评估**阶段则是衡量这些输出与标准答案的匹配程度。这两个过程会被拆分为多个同时运行的“任务”以提高效率，但请注意，如果计算资源有限，这种策略可能会使评测变得更慢。如果需要了解该问题及解决方案，可以参考 [FAQ: 效率](faq.md#效率)。

**可视化**：评估完成后，OpenCompass 将结果整理成易读的表格，并将其保存为 CSV 和 TXT 文件。你也可以激活飞书状态上报功能，此后可以在飞书客户端中及时获得评测状态报告。

接下来，我们将展示 OpenCompass 的基础用法，展示预训练模型 [OPT-125M](https://huggingface.co/facebook/opt-125m) 和 [OPT-350M](https://huggingface.co/facebook/opt-350m) 在 [SIQA](https://huggingface.co/datasets/social_i_qa) 和 [Winograd](https://huggingface.co/datasets/winograd_wsc) 基准任务上的评估。它们的配置文件可以在 [configs/eval_demo.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_demo.py) 中找到。

在运行此实验之前，请确保您已在本地安装了 OpenCompass。这个例子可以在一台 _GTX-1660-6G_ GPU 下成功运行。
对于参数更大的模型，如 Llama-7B，请参考 [configs 目录](https://github.com/open-compass/opencompass/tree/main/configs) 中提供的其他示例。

## 配置评估任务

在 OpenCompass 中，每个评估任务由待评估的模型和数据集组成。评估的入口点是 `run.py`。用户可以通过命令行或配置文件选择要测试的模型和数据集。

`````{tabs}
````{tab} 命令行

用户可以使用 `--models` 和 `--datasets` 结合想测试的模型和数据集。

```bash
python run.py --models hf_opt_125m hf_opt_350m --datasets siqa_gen winograd_ppl
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

````{tab} 命令行（自定义 HF 模型）

对于 HuggingFace 模型，用户可以通过命令行直接设置模型参数，无需额外的配置文件。例如，对于 `facebook/opt-125m` 模型，您可以使用以下命令进行评估：

```bash
python run.py --datasets siqa_gen winograd_ppl \
--hf-path facebook/opt-125m \
--model-kwargs device_map='auto' \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--max-seq-len 2048 \
--max-out-len 100 \
--batch-size 128  \
--num-gpus 1  # 最少需要的 GPU 数量
```

请注意，通过这种方式，OpenCompass 一次只评估一个模型，而其他方式可以一次评估多个模型。

```{caution}
`--num-gpus` 不代表实际用于评估的 GPU 数量，而是该模型所需的最少 GPU 数量。[更多](faq.md#opencompass-如何分配-gpu)
```



:::{dropdown} 更详细的示例
:animate: fade-in-slide-down
```bash
python run.py --datasets siqa_gen winograd_ppl \
--hf-path facebook/opt-125m \  # HuggingFace 模型路径
--tokenizer-path facebook/opt-125m \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
--model-kwargs device_map='auto' \  # 构建模型的参数
--max-seq-len 2048 \  # 模型可以接受的最大序列长度
--max-out-len 100 \  # 生成的最大 token 数
--batch-size 64  \  # 批量大小
--num-gpus 1  # 运行模型所需的 GPU 数量
```
```{seealso}
有关 `run.py` 支持的所有与 HuggingFace 相关的参数，请阅读 [评测任务发起](../user_guides/experimentation.md#评测任务发起)
```
:::


````
````{tab} 配置文件

除了通过命令行配置实验外，OpenCompass 还允许用户在配置文件中编写实验的完整配置，并通过 `run.py` 直接运行它。配置文件是以 Python 格式组织的，并且必须包括 `datasets` 和 `models` 字段。

本次测试配置在 [configs/eval_demo.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_demo.py) 中。此配置通过 [继承机制](../user_guides/config.md#继承机制) 引入所需的数据集和模型配置，并以所需格式组合 `datasets` 和 `models` 字段。

```python
from mmengine.config import read_base

with read_base():
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .datasets.winograd.winograd_ppl import winograd_datasets
    from .models.opt.hf_opt_125m import opt125m
    from .models.opt.hf_opt_350m import opt350m

datasets = [*siqa_datasets, *winograd_datasets]
models = [opt125m, opt350m]
```

运行任务时，我们只需将配置文件的路径传递给 `run.py`：

```bash
python run.py configs/eval_demo.py
```

:::{dropdown} 关于 `models`
:animate: fade-in-slide-down

OpenCompass 提供了一系列预定义的模型配置，位于 `configs/models` 下。以下是与 [opt-350m](https://github.com/open-compass/opencompass/blob/main/configs/models/opt/hf_opt_350m.py)（`configs/models/opt/hf_opt_350m.py`）相关的配置片段：

```python
# 使用 `HuggingFaceCausalLM` 评估由 HuggingFace 的 `AutoModelForCausalLM` 支持的模型
from opencompass.models import HuggingFaceCausalLM

# OPT-350M
opt350m = dict(
       type=HuggingFaceCausalLM,
       # `HuggingFaceCausalLM` 的初始化参数
       path='facebook/opt-350m',
       tokenizer_path='facebook/opt-350m',
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True),
       model_kwargs=dict(device_map='auto'),
       # 下面是所有模型的共同参数，不特定于 HuggingFaceCausalLM
       abbr='opt350m',               # 结果显示的模型缩写
       max_seq_len=2048,             # 整个序列的最大长度
       max_out_len=100,              # 生成的最大 token 数
       batch_size=64,                # 批量大小
       run_cfg=dict(num_gpus=1),     # 该模型所需的 GPU 数量
    )
```

使用配置时，我们可以通过命令行参数 `--models` 指定相关文件，或使用继承机制将模型配置导入到配置文件中的 `models` 列表中。

```{seealso}
有关模型配置的更多信息，请参见 [准备模型](../user_guides/models.md)。
```
:::

:::{dropdown} 关于 `datasets`
:animate: fade-in-slide-down

与模型类似，数据集的配置文件也提供在 `configs/datasets` 下。用户可以在命令行中使用 `--datasets`，或通过继承在配置文件中导入相关配置

下面是来自 `configs/eval_demo.py` 的与数据集相关的配置片段：

```python
from mmengine.config import read_base  # 使用 mmengine.read_base() 读取基本配置

with read_base():
    # 直接从预设的数据集配置中读取所需的数据集配置
    from .datasets.winograd.winograd_ppl import winograd_datasets  # 读取 Winograd 配置，基于 PPL（困惑度）进行评估
    from .datasets.siqa.siqa_gen import siqa_datasets  # 读取 SIQA 配置，基于生成进行评估

datasets = [*siqa_datasets, *winograd_datasets]       # 最终的配置需要包含所需的评估数据集列表 'datasets'
```

数据集配置通常有两种类型：'ppl' 和 'gen'，分别指示使用的评估方法。其中 `ppl` 表示辨别性评估，`gen` 表示生成性评估。

此外，[configs/datasets/collections](https://github.com/open-compass/opencompass/blob/main/configs/datasets/collections) 收录了各种数据集集合，方便进行综合评估。OpenCompass 通常使用 [`base_medium.py`](https://github.com/open-compass/opencompass/blob/main/configs/datasets/collections/base_medium.py) 进行全面的模型测试。要复制结果，只需导入该文件，例如：

```bash
python run.py --models hf_llama_7b --datasets base_medium
```

```{seealso}
您可以从 [配置数据集](../user_guides/datasets.md) 中找到更多信息。
```
:::


````
`````

```{warning}
OpenCompass 通常假定运行环境网络是可用的。如果您遇到网络问题或希望在离线环境中运行 OpenCompass，请参阅 [FAQ - 网络 - Q1](./faq.md#网络) 寻求解决方案。
```

接下来的部分将使用基于配置的方法作为示例来解释其他特征。

## 启动评估

由于 OpenCompass 默认并行启动评估过程，我们可以在第一次运行时以 `--debug` 模式启动评估，并检查是否存在问题。在 `--debug` 模式下，任务将按顺序执行，并实时打印输出。

```bash
python run.py configs/eval_demo.py -w outputs/demo --debug
```

预训练模型 'facebook/opt-350m' 和 'facebook/opt-125m' 将在首次运行期间从 HuggingFace 自动下载。
如果一切正常，您应该看到屏幕上显示 “Starting inference process”：

```bash
[2023-07-12 18:23:55,076] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...
```

然后，您可以按 `ctrl+c` 中断程序，并以正常模式运行以下命令：

```bash
python run.py configs/eval_demo.py -w outputs/demo
```

在正常模式下，评估任务将在后台并行执行，其输出将被重定向到输出目录 `outputs/demo/{TIMESTAMP}`。前端的进度条只指示已完成任务的数量，而不考虑其成功或失败。**任何后端任务失败都只会在终端触发警告消息。**

:::{dropdown} `run.py` 中的更多参数
:animate: fade-in-slide-down
以下是与评估相关的一些参数，可以帮助您根据环境配置更有效的推理任务：

- `-w outputs/demo`：保存评估日志和结果的工作目录。在这种情况下，实验结果将保存到 `outputs/demo/{TIMESTAMP}`。
- `-r`：重用现有的推理结果，并跳过已完成的任务。如果后面跟随时间戳，将重用工作空间路径下该时间戳的结果；否则，将重用指定工作空间路径下的最新结果。
- `--mode all`：指定任务的特定阶段。
  - all：（默认）执行完整评估，包括推理和评估。
  - infer：在每个数据集上执行推理。
  - eval：根据推理结果进行评估。
  - viz：仅显示评估结果。
- `--max-partition-size 40000`：数据集分片大小。一些数据集可能很大，使用此参数可以将它们分成多个子任务以有效利用资源。但是，如果分片过细，由于模型加载时间较长，整体速度可能会变慢。
- `--max-num-workers 32`：并行任务的最大数量。在如 Slurm 之类的分布式环境中，此参数指定提交任务的最大数量。在本地环境中，它指定同时执行的任务的最大数量。请注意，实际的并行任务数量取决于可用的 GPU 资源，可能不等于这个数字。

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
dataset    version    metric    mode      opt350m    opt125m
---------  ---------  --------  ------  ---------  ---------
siqa       e78df3     accuracy  gen         21.55      12.44
winograd   b6c7ed     accuracy  ppl         51.23      49.82
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
│   ├── predictions   # 每个任务的推理结果
│   ├── results       # 每个任务的评估结果
│   └── summary       # 单个实验的汇总评估结果
├── ...
```

打印评测结果的过程可被进一步定制化，用于输出一些数据集的平均分 (例如 MMLU, C-Eval 等)。

关于评测结果输出的更多介绍可阅读 [结果展示](../user_guides/summarizer.md)。

## 更多教程

想要更多了解 OpenCompass, 可以点击下列链接学习。

- [配置数据集](../user_guides/datasets.md)
- [准备模型](../user_guides/models.md)
- [任务运行和监控](../user_guides/experimentation.md)
- [如何调Prompt](../prompt/overview.md)
- [结果展示](../user_guides/summarizer.md)
- [学习配置文件](../user_guides/config.md)
