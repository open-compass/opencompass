import os.path as osp
from typing import Dict, List, Optional

from mmengine.config import Config, ConfigDict

from opencompass.registry import PARTITIONERS
from opencompass.utils import get_infer_output_path

from .base import BasePartitioner


@PARTITIONERS.register_module()
class NaivePartitioner(BasePartitioner):
    """Naive task partitioner. This partitioner will generate a task for each n
    model-dataset pairs.

    Args:
        out_dir (str): The output directory of tasks.
        n (int): The number of model-dataset pairs in each task.
        keep_keys (List[str]): The keys to be kept from the experiment config
            to the task config.
    """

    def __init__(self,
                 out_dir: str,
                 n: int = 1,
                 keep_keys: Optional[List[str]] = None):
        super().__init__(out_dir=out_dir, keep_keys=keep_keys)
        self.n = n

    def partition(self,
                  models: List[ConfigDict],
                  datasets: List[ConfigDict],
                  work_dir: str,
                  out_dir: str,
                  add_cfg: Dict = {}) -> List[Dict]:
        """Partition model-dataset pairs into tasks. Each task is defined as a
        dict and will run independently as a unit. Its structure is as
        follows:

        .. code-block:: python

            {
                'models': [],  # a list of model configs
                'datasets': [[]],  # a nested list of dataset configs, each
                                    list corresponds to a model
                'work_dir': '',  # the work dir
            }

        Args:
            models (List[ConfigDict]): A list of model configs.
            datasets (List[ConfigDict]): A list of dataset configs.
            work_dir (str): The work dir for the task.
            out_dir (str): The full output path for the task, intended for
                Partitioners to check whether the task is finished via the
                existency of result file in this directory.

        Returns:
            List[Dict]: A list of tasks.
        """

        tasks = []
        for model in models:
            chunks = []
            for dataset in datasets:
                filename = get_infer_output_path(model, dataset, out_dir)
                if osp.exists(filename):
                    continue
                chunks.append(dataset)

            for i in range(0, len(chunks), self.n):
                task = Config({
                    'models': [model],
                    'datasets': [chunks[i:i + self.n]],
                    'work_dir': work_dir,
                    **add_cfg
                })
                tasks.append(task)
        return tasks
