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

   - To facilitate the access of your datasets to other users, you need to specify the channels for downloading the datasets in the configuration file. Specifically, you need to first fill in a dataset name given by yourself in the `path` field in the `mydataset_datasets` configuration, and this name will be mapped to the actual download path in the `opencompass/utils/datasets_info.py` file. Here's an example:

   ```python
    mmlu_datasets = [an
        dict(
            ...,
            path='opencompass/mmlu',
            ...,
        )
   ]
   ```

   - Next, you need to create a dictionary key in `opencompass/utils/datasets_info.py` with the same name as the one you provided above. If you have already hosted the dataset on HuggingFace or Modelscope, please add a dictionary key to the `DATASETS_MAPPING` dictionary and fill in the HuggingFace or Modelscope dataset address in the `hf_id` or `ms_id` key, respectively. You can also specify a default local address. Here's an example:

   ```python
   "opencompass/mmlu": {
        "ms_id": "opencompass/mmlu",
        "hf_id": "opencompass/mmlu",
        "local": "./data/mmlu/",
    }
   ```

   - If you wish for the provided dataset to be directly accessible from the OpenCompass OSS repository when used by others, you need to submit the dataset files in the Pull Request phase. We will then transfer the dataset to the OSS on your behalf and create a new dictionary key in the `DATASET_URL`.

   - To ensure the optionality of data sources, you need to improve the method `load` in the dataset script `mydataset.py`. Specifically, you need to implement a functionality to switch among different download sources based on the setting of the environment variable `DATASET_SOURCE`. It should be noted that if the environment variable `DATASET_SOURCE` is not set, the dataset will default to being downloaded from the OSS repository. Here's an example from `opencompass/dataset/cmmlu.py`:

   ```python
    def load(path: str, name: str, **kwargs):
        ...
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            ...
        else:
            ...
        return dataset
   ```

3. After completing the dataset script and config file, you need to register the information of your new dataset in the file `dataset-index.yml` at the main directory, so that it can be added to the dataset statistics list on the OpenCompass website.

   - The keys that need to be filled in include `name`: the name of your dataset, `category`: the category of your dataset, `paper`: the URL of the paper or project, and `configpath`: the path to the dataset config file. Here's an example:

   ```
   - mydataset:
       name: MyDataset
       category: Understanding
       paper: https://arxiv.org/pdf/xxxxxxx
       configpath: opencompass/configs/datasets/MyDataset
   ```

   Detailed dataset configuration files and other required configuration files can be referred to in the [Configuration Files](../user_guides/config.md) tutorial. For guides on launching tasks, please refer to the [Quick Start](../get_started/quick_start.md) tutorial.
