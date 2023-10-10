from itertools import combinations
from typing import Dict, List, Optional, Tuple

from mmengine.config import ConfigDict

from opencompass.registry import PARTITIONERS

from .naive import NaivePartitioner


@PARTITIONERS.register_module()
class SubjectiveNaivePartitioner(NaivePartitioner):
    """Naive task partitioner for subjective evaluation. Compared to
    NaivePartitioner, this partitioner squashes multiple models into a task.

    Args:
        out_dir (str): The output directory of tasks.
        keep_keys (List[str]): The keys to be kept from the experiment config
            to the task config.
    """

    def __init__(self,
                 mode: str,
                 out_dir: str,
                 model_pairs: Optional[List[Tuple]] = None,
                 keep_keys: List[str] = ['eval.runner.task.judge_cfg']):
        super().__init__(out_dir=out_dir, keep_keys=keep_keys)
        assert mode in ['all', 'one_to_n', 'fixed']
        self.mode = mode
        self.model_pairs = model_pairs

    def get_model_combinations(self, models: List[ConfigDict]) -> List:
        if self.mode == 'all':
            return combinations(models, 2)
        elif self.mode == 'one_to_n':
            pass
        elif self.mode == 'fixed':
            pass

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

        models = self.get_model_combinations(models)
        return super().partition(models=models,
                                 datasets=datasets,
                                 work_dir=work_dir,
                                 out_dir=out_dir,
                                 add_cfg=add_cfg)
