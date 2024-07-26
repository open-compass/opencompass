import copy
import math
import os.path as osp
from typing import Dict, List, Optional

import mmengine
from mmengine.config import Config, ConfigDict

from opencompass.registry import PARTITIONERS
from opencompass.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               get_infer_output_path)

from .base import BasePartitioner


@PARTITIONERS.register_module()
class NumWorkerPartitioner(BasePartitioner):
    """Task partitioner based on the pre-defined number of workers.

    Args:
        out_dir (str): The output directory of tasks.
        num_worker (int): The number of workers. default: 8.
        num_split (int): The number of splits for each dataset, set to
            num_worker if not specified. default: None.
        min_task_size (int): The minimum size of a task. default: 16.
        dataset_size_path (str): The path to the dataset size cache file.
        keep_keys (list[str]): The keys to be kept from the experiment config
            to the task config.
    """

    def __init__(self,
                 out_dir: str,
                 num_worker: int = 8,
                 num_split: Optional[int] = None,
                 min_task_size: int = 16,
                 strategy: str = 'heuristic',
                 dataset_size_path: str = '.cache/dataset_size.json',
                 keep_keys: Optional[List[str]] = None):
        super().__init__(out_dir=out_dir, keep_keys=keep_keys)
        if strategy == 'split' and num_worker is not None:
            self.logger.warning('num_worker is ignored with split.')

        self.num_worker = num_worker
        self.num_split = num_split or num_worker
        self.min_task_size = min_task_size
        self.dataset_size_path = dataset_size_path
        assert strategy in ('heuristic', 'split'), \
            f'Unsupported partition strategy: {strategy}. '\
            'Supported strategies are: `heuristic`, `split` .'
        self.strategy = strategy

    def partition(self,
                  model_dataset_combinations: List[Dict[str, List]],
                  work_dir: str,
                  out_dir: str,
                  add_cfg: Dict = {}) -> List[ConfigDict]:

        # intentionally avoid any sort here,
        # for user's abaility to manipulate the order
        tasks = []
        for comb in model_dataset_combinations:
            for model in comb['models']:
                chunks = []
                for dataset in comb['datasets']:
                    filename = get_infer_output_path(model, dataset, out_dir)
                    # skip the task if the task output exists
                    if osp.exists(filename):
                        continue
                    dataset_size = self.get_size(dataset)
                    if self.num_split <= 1:
                        chunks.append(dataset)
                    elif dataset_size <= self.min_task_size:
                        chunks.append(dataset)
                    else:
                        root, ext = osp.splitext(filename)
                        dataset_splits = self.split_dataset(dataset)
                        for i, dataset_split in enumerate(dataset_splits):
                            if not osp.exists(f'{root}_{i}{ext}'):
                                chunks.append(dataset_split)

                if self.strategy == 'heuristic':
                    buckets = [[] for _ in range(self.num_worker)]
                    for i, chunk in enumerate(chunks):
                        buckets[i % self.num_worker].append(chunk)

                    for bucket in buckets:
                        if len(bucket) > 0:
                            tasks.append(
                                Config({
                                    'models': [model],
                                    'datasets': [bucket],
                                    'work_dir': work_dir,
                                    **add_cfg
                                }))
                elif self.strategy == 'split':
                    for dataset in chunks:
                        tasks.append(
                            Config({
                                'models': [model],
                                'datasets': [[dataset]],
                                'work_dir': work_dir,
                                **add_cfg
                            }))
        return tasks

    @property
    def dataset_size(self):
        if not hasattr(self, '_dataset_size'):
            if osp.exists(self.dataset_size_path):
                self._dataset_size = mmengine.load(self.dataset_size_path)
            else:
                self._dataset_size = {}
        return self._dataset_size

    def split_dataset(self, dataset_cfg: ConfigDict) -> List[ConfigDict]:
        """Split dataset into several parts."""
        dataset_size = self.get_size(dataset_cfg)
        split_configs = []
        abbr = dataset_abbr_from_cfg(dataset_cfg)
        # evenly distribute the task
        num_split = self.num_split
        step = max(math.ceil(dataset_size / num_split), self.min_task_size)
        for part, i in enumerate(range(0, dataset_size, step)):
            cfg = copy.deepcopy(dataset_cfg)
            cfg['abbr'] = abbr + f'_{part}'
            test_range = cfg['reader_cfg'].get('test_range', '')
            cfg['reader_cfg']['test_range'] = f'{test_range}[{i}:{i+step}]'
            split_configs.append(cfg)
        return split_configs

    def get_size(self, dataset: ConfigDict) -> int:
        dataset_abbr = dataset_abbr_from_cfg(dataset)

        test_range = dataset.reader_cfg.get('test_range', '')

        if dataset_abbr in self.dataset_size:
            actual_size = eval('len(range(self.dataset_size[dataset_abbr])'
                               f'{test_range})')
            return actual_size

        dataset = build_dataset_from_cfg(dataset)
        self.dataset_size[dataset_abbr] = len(dataset.test)

        mmengine.mkdir_or_exist('.cache/')
        mmengine.dump(self.dataset_size,
                      self.dataset_size_path,
                      indent=4,
                      ensure_ascii=False)

        actual_size = eval('len(range(self.dataset_size[dataset_abbr])'
                           f'{test_range})')
        return actual_size
