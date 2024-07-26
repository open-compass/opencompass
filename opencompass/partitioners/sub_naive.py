# flake8: noqa: E501
import copy
import os.path as osp
from itertools import combinations, product
from typing import Dict, List, Optional, Tuple

from mmengine.config import ConfigDict

from opencompass.registry import PARTITIONERS
from opencompass.utils import (deal_with_judge_model_abbr,
                               get_infer_output_path, model_abbr_from_cfg)

from .naive import NaivePartitioner


def remove_duplicate_pairs(model_combinations):
    # For compare mode, we need to remove redundant pairs first
    combo_dict = {}
    for i, combo in enumerate(model_combinations):
        sorted_names = tuple(sorted((combo[0]['abbr'], combo[1]['abbr'])))
        if sorted_names not in combo_dict:
            combo_dict[sorted_names] = i
    new_model_combinations = [
        model_combinations[i] for i in combo_dict.values()
    ]
    return new_model_combinations


def replicate_tasks_with_judge_models(tasks, judge_models, meta_judge_model):
    # When all tasks are already partitioned, we add judge_models and meta_judge_model as additional args.
    if meta_judge_model:
        replicated_tasks = [[], []]
    else:
        replicated_tasks = []
    for task in tasks:
        replicated_task_dicts = [task.copy() for _ in range(len(judge_models))]
        for idx, replicated_task in enumerate(replicated_task_dicts):
            replicated_task['judge_model'] = judge_models[idx]
        if meta_judge_model:
            meta_task = task.copy()
            meta_task['meta_judge_model'] = meta_judge_model
            meta_task['judge_models'] = judge_models
            replicated_tasks[1].append(meta_task)
            replicated_tasks[0].extend(replicated_task_dicts)
        else:
            replicated_tasks.extend(replicated_task_dicts)
    return replicated_tasks


def remove_already_tasks(tasks, work_dir, meta_judge_model):
    # Check and remove the already finished subjective evaluation tasks
    if isinstance(tasks, list) and len(tasks) != 0 and isinstance(
            tasks[0], list):
        tasks_to_keep = [[], []]
        for i in range(2):
            for task in tasks[i]:
                temp_task = copy.deepcopy(task)
                to_delete_index = [
                ]  # To deal with the situation that the partition strategy is not split, which means that there will be a task contains multi dataset, and when we need to re-start, we need to remove the already done tasks.
                for idx, dataset in enumerate(task['datasets'][0]):
                    if i == 0:
                        filename = get_infer_output_path(
                            deal_with_judge_model_abbr(task['models'][0],
                                                       task['judge_model'],
                                                       False), dataset,
                            osp.join(work_dir, 'results'))
                    else:
                        filename = get_infer_output_path(
                            deal_with_judge_model_abbr(
                                task['models'][0], task['meta_judge_model'],
                                True), dataset, osp.join(work_dir, 'results'))
                    if osp.exists(filename):
                        to_delete_index.append(idx)
                temp_task['datasets'][0] = [
                    temp_task['datasets'][0][j]
                    for j in range(len(temp_task['datasets'][0]))
                    if j not in to_delete_index
                ]
                if len(temp_task['datasets'][0]) != 0:
                    tasks_to_keep[i].append(temp_task)
    else:
        tasks_to_keep = []
        for task in tasks:
            temp_task = copy.deepcopy(task)
            to_delete_index = [
            ]  # To deal with the situation that the partition strategy is not split, which means that there will be a task contains multi dataset, and when we need to re-start, we need to remove the already done tasks.
            for idx, dataset in enumerate(task['datasets'][0]):
                filename = get_infer_output_path(
                    deal_with_judge_model_abbr(task['models'][0],
                                               task['judge_model']), dataset,
                    osp.join(work_dir, 'results'))
                if osp.exists(filename):
                    to_delete_index.append(idx)
            # Remove the already done tasks
            temp_task['datasets'][0] = [
                temp_task['datasets'][0][j]
                for j in range(len(temp_task['datasets'][0]))
                if j not in to_delete_index
            ]
            if len(temp_task['datasets'][0]) != 0:
                tasks_to_keep.append(temp_task)
    return tasks_to_keep


def get_model_combinations(
        mode,
        models: List[ConfigDict],
        base_models: Optional[List[ConfigDict]] = [],
        compare_models: Optional[List[ConfigDict]] = []) -> List:
    if mode == 'allpair':
        assert len(models) > 1
        return combinations(models, 2)
    elif mode == 'm2n':
        assert len(base_models) > 0 and len(compare_models) > 0
        model_combinations = list(product(base_models, compare_models))
        unique_combinations = remove_duplicate_pairs(
            [combo for combo in model_combinations if combo[0] != combo[1]])
        return unique_combinations
    elif mode == 'fixed':
        pass
        return None


@PARTITIONERS.register_module()
class SubjectiveNaivePartitioner(NaivePartitioner):
    """Naive task partitioner for subjective evaluation. Compared to
    NaivePartitioner, this partitioner squashes multiple models into a task.

    Args:
        out_dir (str): The output directory of tasks.
        keep_keys (List[str]): The keys to be kept from the experiment config
            to the task config.
    """

    def __init__(
        self,
        out_dir: str,
        models: Optional[List[ConfigDict]] = [],
        base_models: Optional[List[ConfigDict]] = [],
        compare_models: Optional[List[ConfigDict]] = [],
        judge_models: Optional[List[ConfigDict]] = [],
        meta_judge_model: Optional[ConfigDict] = None,
        model_pairs: Optional[List[Tuple]] = None,
        keep_keys: Optional[List[str]] = None,
    ):
        super().__init__(out_dir=out_dir, keep_keys=keep_keys)

        self.models = models
        self.base_models = base_models
        self.compare_models = compare_models
        self.model_pairs = model_pairs
        self.judge_models = judge_models
        self.meta_judge_model = meta_judge_model

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
        judge_models, meta_judge_model = self.judge_models, self.meta_judge_model
        all_tasks = []
        for dataset in datasets:
            mode = dataset['mode']
            infer_order = dataset.get('infer_order', None)
            assert mode in ['singlescore', 'allpair', 'm2n', 'fixed']
            assert infer_order in ['random', 'double', None]
            if mode == 'singlescore':
                temp_models = models
            else:
                temp_models = get_model_combinations(mode, models,
                                                     dataset['base_models'],
                                                     models)
            model_dataset_combinations = [{
                'models': temp_models,
                'datasets': [dataset]
            }]

            tasks = super().partition(
                model_dataset_combinations=model_dataset_combinations,
                work_dir=work_dir,
                out_dir=out_dir,
                add_cfg=add_cfg)

            # We need to add judge models and meta-judge-model as new tasks
            # When there is no meta-judge-model, we assign all judge models to each tasks
            # When there is a meta-judge-model, we add an additional task stage
            tasks = replicate_tasks_with_judge_models(tasks, judge_models,
                                                      meta_judge_model)

            # We also need to check and remove the already done tasks
            tasks = remove_already_tasks(tasks, work_dir, meta_judge_model)
            if isinstance(tasks, list) and len(tasks) != 0 and isinstance(
                    tasks[0], list):
                # Refer to meta review judge
                for task_stage in tasks:
                    for task in task_stage:
                        task['infer_order'] = infer_order
            else:
                # Refer to just have review judge
                for task in tasks:
                    task['infer_order'] = infer_order
            all_tasks += tasks
        return all_tasks
