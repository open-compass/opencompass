# 学习配置文件

OpenCompass 使用 OpenMMLab 新式风格的配置文件。如果你之前熟悉 OpenMMLab 风格的配置文件，可以直接阅读
[纯 Python 风格的配置文件（Beta）](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html#python-beta)
了解新式配置文件与原配置文件的区别。如果你之前没有接触过 OpenMMLab 风格的配置文件，
下面我将会用一个简单的例子来介绍配置文件的使用。请确保你安装了最新版本的 MMEngine，以支持新式风格的配置文件。

## 基本格式

OpenCompass 的配置文件都是 Python 格式的，遵从基本的 Python 语法，通过定义变量的形式指定每个配置项。
比如在定义模型时，我们使用如下配置：

```python
# model_cfg.py
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        path='huggyllama/llama-7b',
        model_kwargs=dict(device_map='auto'),
        tokenizer_path='huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        max_out_len=50,
        run_cfg=dict(num_gpus=8, num_procs=1),
    )
]
```

当读取配置文件时，使用 MMEngine 中的 `Config.fromfile` 进行解析。

```python
>>> from mmengine.config import Config
>>> cfg = Config.fromfile('./model_cfg.py')
>>> print(cfg.models[0])
{'type': HuggingFaceCausalLM, 'path': 'huggyllama/llama-7b', 'model_kwargs': {'device_map': 'auto'}, ...}
```

## 继承机制

OpenCompass 的配置文件使用了 Python 的 import 机制进行配置文件的继承。需要注意的是，
我们需要在继承配置文件时使用 `read_base` 上下文管理器。

```python
# inherit.py
from mmengine.config import read_base

with read_base():
    from .model_cfg import models  # model_cfg.py 中的 models 被继承到本配置文件
```

使用 `Config.fromfile` 解析配置文件：

```python
>>> from mmengine.config import Config
>>> cfg = Config.fromfile('./inherit.py')
>>> print(cfg.models[0])
{'type': HuggingFaceCausalLM, 'path': 'huggyllama/llama-7b', 'model_kwargs': {'device_map': 'auto'}, ...}
```

## 评测配置文件示例

```python
# configs/llama7b.py
from mmengine.config import read_base

with read_base():
    # 直接从预设数据集配置中读取需要的数据集配置
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets

# 将需要评测的数据集拼接成 datasets 字段
datasets = [*piqa_datasets, *siqa_datasets]

# 使用 HuggingFaceCausalLM 评测 HuggingFace 中 AutoModelForCausalLM 支持的模型
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        # 以下参数为 HuggingFaceCausalLM 的初始化参数
        path='huggyllama/llama-7b',
        tokenizer_path='huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        # 以下参数为各类模型都必须设定的参数，非 HuggingFaceCausalLM 的初始化参数
        abbr='llama-7b',            # 模型简称，用于结果展示
        max_out_len=100,            # 最长生成 token 数
        batch_size=16,              # 批次大小
        run_cfg=dict(num_gpus=1),   # 运行配置，用于指定资源需求
    )
]
```

## 数据集配置文件示例

以上示例配置文件中，我们直接以继承的方式获取了数据集相关的配置。接下来，
我们会以 PIQA 数据集配置文件为示例，展示数据集配置文件中各个字段的含义。
如果你不打算修改模型测试的 prompt，或者添加新的数据集，则可以跳过这一节的介绍。

PIQA 数据集 [配置文件](https://github.com/open-compass/opencompass/blob/main/configs/datasets/piqa/piqa_ppl_1cf9f0.py)
如下，这是一个基于 PPL（困惑度）进行评测的配置，并且不使用上下文学习方法（In-Context Learning）。

```python
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

# 读取配置
# 加载后的数据集通常以字典形式组织样本，分别指定样本中用于组成 prompt 的输入字段，和作为答案的输出字段
piqa_reader_cfg = dict(
    input_columns=['goal', 'sol1', 'sol2'],
    output_column='label',
    test_split='validation',
)

# 推理配置
piqa_infer_cfg = dict(
    # Prompt 生成配置
    prompt_template=dict(
        type=PromptTemplate,
        # Prompt 模板，模板形式与后续指定的 inferencer 类型相匹配
        # 这里为了计算 PPL，需要指定每个答案对应的 Prompt 模板
        template={
            0: 'The following makes sense: \nQ: {goal}\nA: {sol1}\n',
            1: 'The following makes sense: \nQ: {goal}\nA: {sol2}\n'
        }),
    # 上下文样本配置，此处指定 `ZeroRetriever`，即不使用上下文样本
    retriever=dict(type=ZeroRetriever),
    # 推理方式配置
    #   - PPLInferencer 使用 PPL（困惑度）获取答案
    #   - GenInferencer 使用模型的生成结果获取答案
    inferencer=dict(type=PPLInferencer))

# 评估配置，使用 Accuracy 作为评估指标
piqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

# 数据集配置，以上各个变量均为此配置的参数
# 为一个列表，用于指定一个数据集各个评测子集的配置。
piqa_datasets = [
    dict(
        type=HFDataset,
        path='piqa',
        reader_cfg=piqa_reader_cfg,
        infer_cfg=piqa_infer_cfg,
        eval_cfg=piqa_eval_cfg)
]
```

其中 **Prompt 生成配置** 的详细配置方式，可以参见 [Prompt 模板](../prompt/prompt_template.md)。

## 进阶评测配置

在 OpenCompass 中，我们支持了任务划分器（Partitioner）、运行后端（Runner）等配置项，
用于更加灵活、高效的利用计算资源。

默认情况下，我们会使用基于样本数的方式对推理任务进行划分，你可以在启动任务时使用
`--max-partition-size` 指定进行任务划分的样本数阈值。同时，我们默认使用本地资源进行推理和评估任务，
如果你希望使用 Slurm 集群资源，可以在启动任务时使用 `--slurm` 参数和 `--partition` 参数指定 slurm 运行后端。

进一步地，如果以上功能无法满足你的任务划分和运行后端配置需求，你可以在配置文件中进行更详细的配置。
参见[数据分片](./evaluation.md)。
