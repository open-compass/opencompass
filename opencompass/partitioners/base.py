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
        keep_keys (List[str]): The keys to be kept from the experiment config
            to the task config.
    """

    def __init__(self,
                 out_dir: str,
                 keep_keys: List[str] = ['eval.runner.task.judge_cfg']):
        self.logger = get_logger()
        self.out_dir = out_dir
        self.keep_keys = keep_keys

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

        add_cfg = {}
        for k in self.keep_keys:
            try:
                key_chain = k.split('.')
                ori_ptr = cfg
                tgt_ptr = add_cfg
                for key in key_chain[:-1]:
                    ori_ptr = ori_ptr[key]
                    if key not in tgt_ptr:
                        tgt_ptr[key] = {}
                    tgt_ptr = tgt_ptr[key]
                tgt_ptr[key_chain[-1]] = ori_ptr[key_chain[-1]]
            except Exception:
                self.logger.warning(f'Key {k} not found in config, ignored.')

        tasks = self.partition(models,
                               datasets,
                               work_dir,
                               self.out_dir,
                               add_cfg=add_cfg)

        self.logger.info(f'Partitioned into {len(tasks)} tasks.')
        for i, task in enumerate(tasks):
            self.logger.debug(f'Task {i}: {task_abbr_from_cfg(task)}')

        return tasks

    @abstractmethod
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
                **add_cfg  # other keys to be added in the config
            }

        Args:
            models (List[ConfigDict]): A list of model configs.
            datasets (List[ConfigDict]): A list of dataset configs.
            work_dir (str): The work dir for the task.
            out_dir (str): The full output path for the task, intended for
                Partitioners to check whether the task is finished via the
                existency of result file in this directory.
            add_cfg (dict): Other common keys to be added in the task config,
                used to share the same config among tasks. Defaults to {}.

        Returns:
            List[Dict]: A list of tasks.
        """
