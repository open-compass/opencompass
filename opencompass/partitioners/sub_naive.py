from itertools import combinations, product
from typing import Dict, List, Optional, Tuple

from mmengine.config import ConfigDict

from opencompass.registry import PARTITIONERS

from .naive import NaivePartitioner


def remove_duplicate_pairs(model_combinations):
    combo_dict = {}
    for i, combo in enumerate(model_combinations):
        sorted_names = tuple(sorted((combo[0]['abbr'], combo[1]['abbr'])))
        if sorted_names not in combo_dict:
            combo_dict[sorted_names] = i
    new_model_combinations = [
        model_combinations[i] for i in combo_dict.values()
    ]
    return new_model_combinations


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
                 models: Optional[List[ConfigDict]] = [],
                 base_models: Optional[List[ConfigDict]] = [],
                 compare_models: Optional[List[ConfigDict]] = [],
                 model_pairs: Optional[List[Tuple]] = None,
                 keep_keys: Optional[List[str]] = None):
        super().__init__(out_dir=out_dir, keep_keys=keep_keys)
        assert mode in ['singlescore', 'allpair', 'm2n', 'fixed']
        self.mode = mode
        self.models = models
        self.base_models = base_models
        self.compare_models = compare_models
        self.model_pairs = model_pairs

    def get_model_combinations(
            self,
            models: List[ConfigDict],
            base_models: Optional[List[ConfigDict]] = [],
            compare_models: Optional[List[ConfigDict]] = []) -> List:
        if self.mode == 'allpair':
            assert len(models) > 1
            return combinations(models, 2)
        elif self.mode == 'm2n':
            assert len(base_models) > 0 and len(compare_models) > 0
            model_combinations = list(product(base_models, compare_models))
            unique_combinations = remove_duplicate_pairs([
                combo for combo in model_combinations if combo[0] != combo[1]
            ])
            return unique_combinations
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
        models = self.models if self.models != [] else models
        base_models, compare_models = self.base_models, self.compare_models
        if self.mode == 'singlescore':
            models = models
        else:
            models = self.get_model_combinations(models, base_models,
                                                 compare_models)
        model_dataset_combinations = [{'models': models, 'datasets': datasets}]
        return super().partition(
            model_dataset_combinations=model_dataset_combinations,
            work_dir=work_dir,
            out_dir=out_dir,
            add_cfg=add_cfg)
