# 支持新数据集

尽管 OpenCompass 已经包含了大多数常用数据集，用户在支持新数据集的时候需要完成以下几个步骤：

1. 在 `opencompass/datasets` 文件夹新增数据集脚本 `mydataset.py`, 该脚本需要包含：

   - 数据集及其加载方式，需要定义一个 `MyDataset` 类，实现数据集加载方法 `load`，该方法为静态方法，需要返回 `datasets.Dataset` 或 `datasets.DatasetDict` 类型的数据。这里我们使用 Hugging Face dataset 作为数据集的统一接口，避免引入额外的逻辑。如果返回 `Dataset`，OpenCompass 会将其同时作为内部的 `train` 和 `test` split；如果返回 `DatasetDict`，可以在 `reader_cfg` 中通过 `train_split` 和 `test_split` 指定实际使用的 split。具体示例如下：

   ```python
   from typing import Union

   import datasets
   from opencompass.registry import LOAD_DATASET

   from .base import BaseDataset

   @LOAD_DATASET.register_module()
   class MyDataset(BaseDataset):

       @staticmethod
       def load(**kwargs) -> Union[datasets.Dataset, datasets.DatasetDict]:
           pass
   ```

   - （可选）如果 OpenCompass 已有的评测器不能满足需要，可以实现并注册自定义 Evaluator，具体方法参阅[新增评测器、后处理器与汇总器](new_evaluator_and_summarizer.md#evaluator)。

   - （可选）如果 OpenCompass 已有的后处理方法不能满足需要，需要用户定义 `mydataset_postprocess` 方法，根据输入的字符串得到相应后处理的结果。如果希望通过注册名复用该后处理器，需要注册到 `TEXT_POSTPROCESSORS`。具体示例如下：

   ```python
   from opencompass.registry import TEXT_POSTPROCESSORS

   @TEXT_POSTPROCESSORS.register_module('mydataset')
   def mydataset_postprocess(text: str) -> str:
       pass
   ```

   新增数据集脚本后，需要确保相关类和函数能被配置文件导入。如果希望使用 `from opencompass.datasets import ...` 的写法，需要在 `opencompass/datasets/__init__.py` 中导入新模块；也可以在配置文件中直接从具体模块导入，例如 `from opencompass.datasets.mydataset import MyDataset`。

2. 在定义好数据集加载、评测以及数据后处理等方法之后，需要在配置文件中新增以下配置：

   ```python
   from opencompass.datasets import MyDataset, MyDatasetEvaluator, mydataset_postprocess

   mydataset_eval_cfg = dict(
       evaluator=dict(type=MyDatasetEvaluator),
       pred_postprocessor=dict(type=mydataset_postprocess))

   mydataset_datasets = [
       dict(
           type=MyDataset,
           ...,
           reader_cfg=...,
           infer_cfg=...,
           eval_cfg=mydataset_eval_cfg)
   ]
   ```

   - 为了使用户提供的数据集能够被其他使用者更方便地获取，需要用户在配置文件中给出数据集路径。`path` 字段可以填写本地路径，也可以填写一个逻辑数据集名称；后者会通过 `opencompass/utils/datasets_info.py` 中的映射解析为实际数据源。具体示例如下：

   ```python
    mmlu_datasets = [
        dict(
            ...,
            path='opencompass/mmlu',
            ...,
        )
   ]
   ```

   - 接着，需要在 `opencompass/utils/datasets_info.py` 中创建对应名称的字典字段。如果用户已将数据集托管到 Hugging Face 或 ModelScope，那么请在 `DATASETS_MAPPING` 字典中添加对应名称的字段，并将对应的 Hugging Face 或 ModelScope 数据集地址填入 `hf_id` 和 `ms_id`；另外，还允许指定一个默认的 `local` 地址。具体示例如下：

   ```python
   "opencompass/mmlu": {
        "ms_id": "opencompass/mmlu",
        "hf_id": "opencompass/mmlu",
        "local": "./data/mmlu/",
    }
   ```

   - 如果希望提供的数据集在其他用户使用时能够通过 OpenCompass 官方 OSS 获取，则需要在 Pull Request 阶段向我们提交数据集文件，我们将代为传输数据集至 OSS，并在 `DATASETS_URL` 新建字段。

   - 为了确保数据来源的可选择性，用户需要根据所提供数据集的下载路径类型来完善数据集脚本 `mydataset.py` 中的 `load` 方法。通常先调用 `get_data_path(path)` 解析路径：当 `DATASET_SOURCE=ModelScope` 时会使用 `ms_id`；当 `DATASET_SOURCE=HF` 时会使用 `hf_id`；未设置 `DATASET_SOURCE` 时，当前实现会优先使用 `local` 字段对应的本地路径，并结合 `COMPASS_DATA_CACHE` 查找缓存，路径不存在时才会根据 `DATASETS_URL` 尝试下载。若不同数据源返回的数据格式不同，需要在 `load` 中做相应适配。`opencompass/datasets/cmmlu.py` 中的具体示例如下：

   ```python
    def load(path: str, name: str, **kwargs):
        ...
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            ...
        else:
            ...
        return dataset
   ```

3. 在完成数据集脚本和配置文件的构建后，需要在 OpenCompass 主目录下的 `dataset-index.yml` 配置文件中登记新数据集的相关信息，以使其加入 OpenCompass 官网 Doc 的数据集统计列表中。

   - 需要填写的字段包括数据集名称 `name`、数据集类型 `category`、原文或项目地址 `paper`、数据集配置文件路径 `configpath`，以及 LLM Judge 配置文件路径 `configpath_llmjudge`。如果没有对应的 LLM Judge 配置，可以将 `configpath_llmjudge` 置为空字符串。具体示例如下：

   ```
   - mydataset:
       name: MyDataset
       category: Understanding
       paper: https://arxiv.org/pdf/xxxxxxx
       configpath: opencompass/configs/datasets/MyDataset
       configpath_llmjudge: ''
   ```

详细的数据集配置文件以及其他需要的配置文件可以参考[配置文件](../user_guides/config.md)教程，启动任务相关的教程可以参考[快速开始](../get_started/quick_start.md)教程。
