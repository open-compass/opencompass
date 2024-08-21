# flake8: noqa: E501
import inspect
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional

from mmengine.config import ConfigDict

from opencompass.utils import (dataset_abbr_from_cfg, get_logger,
                               model_abbr_from_cfg, task_abbr_from_cfg)


class BasePartitioner:
    """Base class for partitioners. A partitioner is responsible for
    partitioning the config into tasks.

    Args:
        out_dir (str): The output directory of tasks.
        keep_keys (Optional[List[str]], optional): The keys to be kept from the
            experiment config to the task config. Defaults to None. If None,
            the following keys will be kept:

            - eval.runner.task.judge_cfg
            - eval.runner.task.dump_details
    """

    def __init__(self, out_dir: str, keep_keys: Optional[List[str]] = None):
        self.logger = get_logger()
        self.out_dir = out_dir
        if keep_keys is None:
            self.keep_keys = [
                'eval.runner.task.judge_cfg',
                'eval.runner.task.dump_details',
                'eval.given_pred',
                'eval.runner.task.cal_extract_rate',
            ]
        else:
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
                self.logger.debug(f'Key {k} not found in config, ignored.')
        self.logger.debug(f'Additional config: {add_cfg}')

        model_and_dataset_args = self.parse_model_dataset_args(cfg)

        tasks = self.partition(**model_and_dataset_args,
                               work_dir=work_dir,
                               out_dir=self.out_dir,
                               add_cfg=add_cfg)
        if isinstance(tasks, list) and len(tasks) != 0 and isinstance(
                tasks[0], list):
            self.logger.info(
                f'Now we are in the subjective evluation! Partitioned into 2 stages and {len(tasks[0])} tasks in first stage, {len(tasks[1])} tasks in second stage.'
            )
            cnt = 0
            for task_part in tasks:
                for task in task_part:
                    self.logger.debug(
                        f'Task {cnt}: {task_abbr_from_cfg(task)}')
                    cnt += 1
        else:
            self.logger.info(f'Partitioned into {len(tasks)} tasks.')
            for i, task in enumerate(tasks):
                self.logger.debug(f'Task {i}: {task_abbr_from_cfg(task)}')
        return tasks

    def parse_model_dataset_args(self, cfg: ConfigDict):
        models = cfg['models']
        datasets = cfg['datasets']

        sig = inspect.signature(self.partition)
        if 'model_dataset_combinations' in sig.parameters:
            combs = cfg.get('model_dataset_combinations', None)
            if combs is None:
                combs = [{'models': models, 'datasets': datasets}]
            else:
                # sanity check
                model_abbrs = [model_abbr_from_cfg(model) for model in models]
                dataset_abbrs = [
                    dataset_abbr_from_cfg(dataset) for dataset in datasets
                ]
                for comb in combs:
                    for model in comb['models']:
                        if model_abbr_from_cfg(model) not in model_abbrs:
                            raise ValueError(
                                f'Model {model_abbr_from_cfg(model)} '
                                'not found in config.')
                    for dataset in comb['datasets']:
                        if dataset_abbr_from_cfg(dataset) not in dataset_abbrs:
                            raise ValueError(
                                f'Dataset {dataset_abbr_from_cfg(dataset)} '
                                'not found in config.')
            used_kwargs = {'model_dataset_combinations': combs}
        else:
            if cfg.get('model_dataset_combinations', None) is not None:
                self.logger.warning(
                    'model_dataset_combinations is not supported by '
                    f'{self.__class__.__name__}. Ignored.')
            used_kwargs = {'models': models, 'datasets': datasets}
        return used_kwargs

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
