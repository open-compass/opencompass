# 支持新数据集

尽管 OpenCompass 已经包含了大多数常用数据集，用户在支持新数据集的时候需要完成以下几个步骤：

1. 在 `opencompass/datasets` 文件夹新增数据集脚本 `mydataset.py`, 该脚本需要包含：

   - 数据集及其加载方式，需要定义一个 `MyDataset` 类，实现数据集加载方法 `load`，该方法为静态方法，需要返回 `datasets.Dataset` 类型的数据。这里我们使用 huggingface dataset 作为数据集的统一接口，避免引入额外的逻辑。具体示例如下：

   ```python
   import datasets
   from .base import BaseDataset

   class MyDataset(BaseDataset):

       @staticmethod
       def load(**kwargs) -> datasets.Dataset:
           pass
   ```

   - （可选）如果 OpenCompass 已有的评测器不能满足需要，需要用户定义 `MyDatasetlEvaluator` 类，实现评分方法 `score`，需要根据输入的 `predictions` 和 `references` 列表，得到需要的字典。由于一个数据集可能存在多种 metric，需要返回一个 metrics 以及对应 scores 的相关字典。具体示例如下：

   ```python
   from opencompass.openicl.icl_evaluator import BaseEvaluator

   class MyDatasetlEvaluator(BaseEvaluator):

       def score(self, predictions: List, references: List) -> dict:
           pass

   ```

   - （可选）如果 OpenCompass 已有的后处理方法不能满足需要，需要用户定义 `mydataset_postprocess` 方法，根据输入的字符串得到相应后处理的结果。具体示例如下：

   ```python
   def mydataset_postprocess(text: str) -> str:
       pass
   ```

2. 在定义好数据集加载、评测以及数据后处理等方法之后，需要在配置文件中新增以下配置：

   ```python
   from opencompass.datasets import MyDataset, MyDatasetlEvaluator, mydataset_postprocess

   mydataset_eval_cfg = dict(
       evaluator=dict(type=MyDatasetlEvaluator),
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

   - 为了使用户提供的数据集能够被其他使用者更方便的获取，需要用户在配置文件中给出下载数据集的渠道。具体的方式是首先在`mydataset_datasets`配置中的`path`字段填写用户指定的数据集名称，该名称将以mapping的方式映射到`opencompass/utils/datasets_info.py`中的实际下载路径。具体示例如下：

   ```python
    mmlu_datasets = [
        dict(
            ...,
            path='opencompass/mmlu',
            ...,
        )
   ]
   ```

   - 接着，需要在`opencompass/utils/datasets_info.py`中创建对应名称的字典字段。如果用户已将数据集托管到huggingface或modelscope，那么请在`DATASETS_MAPPING`字典中添加对应名称的字段，并将对应的huggingface或modelscope数据集地址填入`ms_id`和`hf_id`；另外，还允许指定一个默认的`local`地址。具体示例如下：

   ```python
   "opencompass/mmlu": {
        "ms_id": "opencompass/mmlu",
        "hf_id": "opencompass/mmlu",
        "local": "./data/mmlu/",
    }
   ```

   - 如果希望提供的数据集在其他用户使用时能够直接从OpenCompass官方的OSS仓库获取，则需要在Pull Request阶段向我们提交数据集文件，我们将代为传输数据集至OSS，并在`DATASET_URL`新建字段。

   - 为了确保数据来源的可选择性，用户需要根据所提供数据集的下载路径类型来完善数据集脚本`mydataset.py`中的`load`方法的功能。具体而言，需要用户实现根据环境变量`DATASET_SOURCE`的不同设置来切换不同的下载数据源的功能。需要注意的是，若未设置`DATASET_SOURCE`的值，将默认从OSS仓库下载数据。`opencompass/dataset/cmmlu.py`中的具体示例如下：

   ```python
    def load(path: str, name: str, **kwargs):
        ...
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            ...
        else:
            ...
        return dataset
   ```

3. 在完成数据集脚本和配置文件的构建后，需要在OpenCompass主目录下的`dataset-index.yml`配置文件中登记新数据集的相关信息，以使其加入OpenCompass官网Doc的数据集统计列表中。

   - 需要填写的字段包括数据集名称`name`、数据集类型`category`、原文或项目地址`paper`、以及数据集配置文件的路径`configpath`。具体示例如下：

   ```
   - mydataset:
       name: MyDataset
       category: Understanding
       paper: https://arxiv.org/pdf/xxxxxxx
       configpath: opencompass/configs/datasets/MyDataset
   ```

详细的数据集配置文件以及其他需要的配置文件可以参考[配置文件](../user_guides/config.md)教程，启动任务相关的教程可以参考[快速开始](../get_started/quick_start.md)教程。
