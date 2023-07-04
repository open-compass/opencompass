from abc import abstractmethod
from copy import deepcopy
from typing import Dict, List

from mmengine.config import ConfigDict

from opencompass.utils import get_logger, task_abbr_from_cfg


class BasePartitioner:
    """Base class for partitioners. A partitioner is responsible for
    partitioning the config into tasks.

    Args:
        out_dir (str): The output directory of tasks.
    """

    def __init__(self, out_dir: str):
        self.logger = get_logger()
        self.out_dir = out_dir

    def __call__(self, cfg: ConfigDict) -> List[Dict]:
        """Generate tasks from config. Each task is defined as a
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
            cfg (ConfigDict): The config dict, containing "models", "dataset"
                and "work_dir" keys.

        Returns:
            List[Dict]: A list of tasks.
        """
        cfg = deepcopy(cfg)
        models = cfg['models']
        datasets = cfg['datasets']
        work_dir = cfg['work_dir']

        tasks = self.partition(models, datasets, work_dir, self.out_dir)

        self.logger.info(f'Partitioned into {len(tasks)} tasks.')
        for i, task in enumerate(tasks):
            self.logger.debug(f'Task {i}: {task_abbr_from_cfg(task)}')

        return tasks

    @abstractmethod
    def partition(self, models: List[ConfigDict], datasets: List[ConfigDict],
                  work_dir: str, out_dir: str) -> List[Dict]:
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
