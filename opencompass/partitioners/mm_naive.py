from copy import deepcopy
from typing import Dict, List

from mmengine.config import Config, ConfigDict

from opencompass.registry import PARTITIONERS

from .base import BasePartitioner


@PARTITIONERS.register_module()
class MultimodalNaivePartitioner(BasePartitioner):
    """Multimodal naive task partitioner.

    This partitioner will generate a task for each
    model-dataset-evaluator pair.

    Args:
        config (ConfigDict): The full config dict.
    """

    def partition(self, models: List[ConfigDict], datasets: List[ConfigDict],
                  evaluators: List[ConfigDict], load_froms: List[ConfigDict],
                  work_dir: str, num_gpus: int, num_procs: int,
                  launcher: str) -> List[Dict]:
        """Partition model-dataset pairs into tasks. Each task is defined as a
        dict and will run independently as a unit. Its structure is as follows:

        .. code-block:: python

            {
                'models': [],  # a list of model configs
                'datasets': [],  # a list of dataset configs
                'evaluators': [], # a list of evaluator configs
                'load_froms': [], # a list of load_from paths
                'work_dir': '',  # the work dir
                'num_gpus': int, # integer, number of gpus for each task
                'num_procs': int, # integer, number of gpus on single machine
                'launcher': str, # string, how to launch distributed training
            }

        Args:
            models (List[ConfigDict]): A list of model configs.
            datasets (List[ConfigDict]): A list of dataset configs.
            evaluators (List[ConfigDict]): A list of evaluator configs.
            load_froms (List[ConfigDict]): A list of load_from paths.
            work_dir (str): The work dir for the task.
            num_gpus (int): Number of gpus for each task.
            num_procs (int): Number of gpus on single machine.
            launcher (str): How to launch distributed training.
                Only `slurm`, `pytorch` and `mpi` are available.

        Returns:
            List[Dict]: A list of tasks.
        """

        tasks = []
        for model, dataset, evaluator, load_from in zip(
                models, datasets, evaluators, load_froms):
            task = Config({
                'model': model,
                'dataset': dataset,
                'evaluator': evaluator,
                'load_from': load_from,
                'work_dir': work_dir,
                'num_gpus': num_gpus,
                'num_procs': num_procs,
                'launcher': launcher
            })
            tasks.append(task)

        return tasks

    def __call__(self, cfg: ConfigDict) -> List[Dict]:
        """Generate tasks from config. Each task is defined as a
        dict and will run independently as a unit. Its structure is as
        follows:

        .. code-block:: python

            {
                'models': [],  # a list of model configs
                'datasets': [],  # a list of dataset configs
                'evaluators': [], # a list of evaluator configs
                'load_froms': [], # a list of load_from paths
                'work_dir': '',  # the work dir
                'num_gpus': int, # integer, number of gpus for each task
                'num_procs': int, # integer, number of gpus on single machine
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
        evaluators = cfg['evaluators']
        load_froms = cfg['load_froms']
        work_dir = cfg['work_dir']
        num_gpus = cfg['num_gpus']
        num_procs = cfg['num_procs']
        launcher = cfg['launcher']

        tasks = self.partition(models, datasets, evaluators, load_froms,
                               work_dir, num_gpus, num_procs, launcher)

        self.logger.info(f'Partitioned into {len(tasks)} tasks.')
        for i, task in enumerate(tasks):
            model_name = task['model']['type']
            dataset_name = task['dataset']['dataset']['type']
            evaluator_name = task['evaluator'][0]['type']
            self.logger.debug(
                f'Task {i}: {model_name}-{dataset_name}-{evaluator_name}')

        return tasks
