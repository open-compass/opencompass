# 代码评测教程

这里以 `humaneval` 和 `mbpp` 为例，主要介绍如何评测模型的代码能力。

## pass@1

如果只需要生成单条回复来评测pass@1的性能，可以直接使用[configs/datasets/humaneval/humaneval_gen_8e312c.py](https://github.com/open-compass/opencompass/blob/main/configs/datasets/humaneval/humaneval_gen_8e312c.py) 和 [configs/datasets/mbpp/deprecated_mbpp_gen_1e1056.py](https://github.com/open-compass/opencompass/blob/main/configs/datasets/mbpp/deprecated_mbpp_gen_1e1056.py) 并参考通用的[快速上手教程](../get_started/quick_start.md)即可。

如果要进行多语言评测，可以参考[多语言代码评测教程](./code_eval_service.md)。

## pass@k

如果对于单个example需要生成多条回复来评测pass@k的性能，需要参考以下两种情况。这里以10回复为例子：

### 通常情况

对于绝大多数模型来说，模型支持HF的generation中带有`num_return_sequences` 参数，我们可以直接使用来获取多回复。可以参考以下配置文件。

```python
from opencompass.datasets import MBPPDatasetV2, MBPPPassKEvaluator

with read_base():
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets

mbpp_datasets[0]['type'] = MBPPDatasetV2
mbpp_datasets[0]['eval_cfg']['evaluator']['type'] = MBPPPassKEvaluator
mbpp_datasets[0]['reader_cfg']['output_column'] = 'test_column'

datasets = []
datasets += humaneval_datasets
datasets += mbpp_datasets

models = [
    dict(
        type=HuggingFaceCausalLM,
        ...,
        generation_kwargs=dict(
            num_return_sequences=10,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        ),
        ...,
    )
]
```

对于 `mbpp`，在数据集和评测上需要有新的变更，所以同步修改`type`, `eval_cfg.evaluator.type`, `reader_cfg.output_column` 字段来适应新的需求。

另外我们需要模型的回复有随机性，同步需要设置`generation_kwargs`参数。这里注意要设置`num_return_sequences`得到回复数。

注意：`num_return_sequences` 必须大于等于k，本身pass@k是计算的概率估计。

具体可以参考以下配置文件
[configs/eval_code_passk.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_code_passk.py)

### 模型不支持多回复

适用于一些没有设计好的API以及功能缺失的HF模型。这个时候我们需要重复构造数据集来达到多回复的效果。这里可以参考以下配置文件。

```python
from opencompass.datasets import MBPPDatasetV2, MBPPPassKEvaluator

with read_base():
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets

humaneval_datasets[0]['abbr'] = 'openai_humaneval_pass10'
humaneval_datasets[0]['num_repeats'] = 10
mbpp_datasets[0]['abbr'] = 'mbpp_pass10'
mbpp_datasets[0]['num_repeats'] = 10
mbpp_datasets[0]['type'] = MBPPDatasetV2
mbpp_datasets[0]['eval_cfg']['evaluator']['type'] = MBPPPassKEvaluator
mbpp_datasets[0]['reader_cfg']['output_column'] = 'test_column'

datasets = []
datasets += humaneval_datasets
datasets += mbpp_datasets

models = [
    dict(
        type=HuggingFaceCausalLM,
        ...,
        generation_kwargs=dict(
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        ),
        ...,
    )
]
```

由于数据集的prompt并没有修改，我们需要替换对应的字段来达到数据集重复的目的。
需要修改以下字段：

- `num_repeats`: 数据集重复的次数
- `abbr`: 数据集的缩写最好随着重复次数一并修改，因为数据集数量会发生变化，防止与`.cache/dataset_size.json` 中的数值出现差异导致一些潜在的问题。

对于 `mbpp`，同样修改`type`, `eval_cfg.evaluator.type`, `reader_cfg.output_column` 字段。

另外我们需要模型的回复有随机性，同步需要设置`generation_kwargs`参数。

具体可以参考以下配置文件
[configs/eval_code_passk_repeat_dataset.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_code_passk_repeat_dataset.py)
