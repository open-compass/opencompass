# Add a dataset

Although OpenCompass has already included most commonly used datasets, users need to follow the steps below to support a new dataset if wanted:

1. Add a dataset script `mydataset.py` to the `opencompass/datasets` folder. This script should include:

   - The dataset and its loading method. Define a `MyDataset` class that implements the data loading method `load` as a static method. This method should return data of type `datasets.Dataset`. We use the Hugging Face dataset as the unified interface for datasets to avoid introducing additional logic. Here's an example:

   ```python
   import datasets
   from .base import BaseDataset

   class MyDataset(BaseDataset):

       @staticmethod
       def load(**kwargs) -> datasets.Dataset:
           pass
   ```

   - (Optional) If the existing evaluators in OpenCompass do not meet your needs, you need to define a `MyDatasetEvaluator` class that implements the scoring method `score`. This method should take `predictions` and `references` as input and return the desired dictionary. Since a dataset may have multiple metrics, the method should return a dictionary containing the metrics and their corresponding scores. Here's an example:

   ```python
   from opencompass.openicl.icl_evaluator import BaseEvaluator

   class MyDatasetEvaluator(BaseEvaluator):

       def score(self, predictions: List, references: List) -> dict:
           pass
   ```

   - (Optional) If the existing postprocessors in OpenCompass do not meet your needs, you need to define the `mydataset_postprocess` method. This method takes an input string and returns the corresponding postprocessed result string. Here's an example:

   ```python
   def mydataset_postprocess(text: str) -> str:
       pass
   ```

2. After defining the dataset loading, data postprocessing, and evaluator methods, you need to add the following configurations to the configuration file:

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

   Detailed dataset configuration files and other required configuration files can be referred to in the [Configuration Files](../user_guides/config.md) tutorial. For guides on launching tasks, please refer to the [Quick Start](../get_started/quick_start.md) tutorial.
