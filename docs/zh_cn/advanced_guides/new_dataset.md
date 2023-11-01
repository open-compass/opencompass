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

   详细的数据集配置文件以及其他需要的配置文件可以参考[配置文件](../user_guides/config.md)教程，启动任务相关的教程可以参考[快速开始](../get_started/quick_start.md)教程。
