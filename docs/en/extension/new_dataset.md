# Add a dataset

Although OpenCompass has already included most commonly used datasets, users need to follow the steps below to support a new dataset if wanted:

1. Add a dataset script `mydataset.py` to the `opencompass/datasets` folder. This script should include:

   - The dataset and its loading method. Define a `MyDataset` class that implements the data loading method `load` as a static method. This method should return data of type `datasets.Dataset` or `datasets.DatasetDict`. We use the Hugging Face dataset as the unified interface for datasets to avoid introducing additional logic. If `load` returns a `Dataset`, OpenCompass will use it as both internal `train` and `test` splits. If it returns a `DatasetDict`, you can specify the actual splits with `train_split` and `test_split` in `reader_cfg`. Here's an example:

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

   - (Optional) If the existing evaluators in OpenCompass do not meet your needs, you can implement and register a custom Evaluator. See [Adding Postprocessors, Evaluators, and Summarizers](new_evaluator_and_summarizer.md#evaluator) for details.

   - (Optional) If the existing postprocessors in OpenCompass do not meet your needs, you need to define the `mydataset_postprocess` method. This method takes an input string and returns the corresponding postprocessed result string. If you want to reuse the postprocessor by a registry name, register it to `TEXT_POSTPROCESSORS`. Here's an example:

   ```python
   from opencompass.registry import TEXT_POSTPROCESSORS

   @TEXT_POSTPROCESSORS.register_module('mydataset')
   def mydataset_postprocess(text: str) -> str:
       pass
   ```

   After adding the dataset script, make sure the related classes and functions can be imported by the config file. If you want to use `from opencompass.datasets import ...`, import the new module in `opencompass/datasets/__init__.py`; alternatively, import directly from the concrete module in the config file, for example `from opencompass.datasets.mydataset import MyDataset`.

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

   - To make your dataset easier for other users to access, specify the dataset path in the configuration file. The `path` field can be a local path or a logical dataset name. A logical dataset name is resolved through the mapping in `opencompass/utils/datasets_info.py`. Here's an example:

   ```python
    mmlu_datasets = [
        dict(
            ...,
            path='opencompass/mmlu',
            ...,
        )
   ]
   ```

   - Next, you need to create a dictionary key in `opencompass/utils/datasets_info.py` with the same name as the one you provided above. If you have already hosted the dataset on Hugging Face or ModelScope, please add a dictionary key to the `DATASETS_MAPPING` dictionary and fill in the Hugging Face or ModelScope dataset address in the `hf_id` or `ms_id` key, respectively. You can also specify a default `local` address. Here's an example:

   ```python
   "opencompass/mmlu": {
        "ms_id": "opencompass/mmlu",
        "hf_id": "opencompass/mmlu",
        "local": "./data/mmlu/",
    }
   ```

   - If you wish for the provided dataset to be accessible through the OpenCompass OSS repository when used by others, you need to submit the dataset files in the Pull Request phase. We will then transfer the dataset to the OSS on your behalf and create a new dictionary key in `DATASETS_URL`.

   - To keep data sources selectable, implement the `load` method in `mydataset.py` according to the path type you provide. Usually, call `get_data_path(path)` first to resolve the path: when `DATASET_SOURCE=ModelScope`, it uses `ms_id`; when `DATASET_SOURCE=HF`, it uses `hf_id`; when `DATASET_SOURCE` is not set, the current implementation first uses the local path from the `local` field and combines it with `COMPASS_DATA_CACHE` to look for cached data. It only tries to download through `DATASETS_URL` when the local path does not exist. If different data sources return different data formats, adapt them in `load`. Here's an example from `opencompass/datasets/cmmlu.py`:

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

   - The keys that need to be filled in include `name`: the name of your dataset, `category`: the category of your dataset, `paper`: the URL of the paper or project, `configpath`: the path to the dataset config file, and `configpath_llmjudge`: the path to the LLM Judge config file. If no LLM Judge config is available, set `configpath_llmjudge` to an empty string. Here's an example:

   ```
   - mydataset:
       name: MyDataset
       category: Understanding
       paper: https://arxiv.org/pdf/xxxxxxx
       configpath: opencompass/configs/datasets/MyDataset
       configpath_llmjudge: ''
   ```

   Detailed dataset configuration files and other required configuration files can be referred to in the [Configuration Files](../user_guides/config.md) tutorial. For guides on launching tasks, please refer to the [Quick Start](../get_started/quick_start.md) tutorial.
