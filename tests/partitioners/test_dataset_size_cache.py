import copy
import json
import os.path as osp
import tempfile
import unittest

from datasets import Dataset, DatasetDict
from mmengine.config import ConfigDict

from opencompass.openicl.icl_dataset_reader import DatasetReader
from opencompass.partitioners import NumWorkerPartitioner, SizePartitioner
from opencompass.utils import build_dataset_from_cfg


class ReaderDataset:
    """Small local dataset using the same reader contract as BaseDataset."""

    def __init__(self, reader_cfg, abbr):
        data = DatasetDict({
            'train':
            Dataset.from_dict({'index': list(range(8))}),
            'test':
            Dataset.from_dict({'index': list(range(64))}),
            'validation':
            Dataset.from_dict({'index': list(range(32))}),
        })
        self.reader = DatasetReader(data, **reader_cfg)
        self.test = self.reader.dataset['test']


class TestDatasetSizeCache(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def make_config(self, test_range):
        return ConfigDict(
            type=ReaderDataset,
            abbr='cache-test',
            reader_cfg=dict(input_columns=['index'],
                            output_column=None,
                            test_split='validation',
                            test_range=test_range,
                            train_range='[:2]'),
            infer_cfg=dict(prompt_template=dict(template='{index}')))

    def make_partitioner(self, cls, strategy='split', **kwargs):
        options = dict(out_dir=self.temp_dir.name,
                       dataset_size_path=osp.join(self.temp_dir.name,
                                                  'size.json'),
                       strategy=strategy)
        if cls is SizePartitioner:
            options.update(max_task_size=4, gen_task_coef=1)
        else:
            options.update(num_worker=2, num_split=2, min_task_size=2)
        return cls(**options, **kwargs)

    def size(self, partitioner, cfg):
        if isinstance(partitioner, SizePartitioner):
            return partitioner.get_cost(cfg)
        return partitioner.get_size(cfg)

    def test_cold_and_warm_sizes_match_reader(self):
        for cls in (SizePartitioner, NumWorkerPartitioner):
            for selection in ('[4:20]', '[::2]', '[4:24][::2]', '[-8:]', ''):
                with self.subTest(partitioner=cls, selection=selection):
                    partitioner = self.make_partitioner(cls)
                    partitioner._dataset_size = {}
                    cfg = self.make_config(selection)
                    original = copy.deepcopy(cfg)
                    expected = len(build_dataset_from_cfg(cfg).test)
                    self.assertEqual(self.size(partitioner, cfg), expected)
                    self.assertEqual(cfg, original)
                    with open(partitioner.dataset_size_path) as cache_file:
                        self.assertEqual(json.load(cache_file),
                                         {'cache-test': 32})
                    warm = self.make_partitioner(cls)
                    self.assertEqual(self.size(warm, cfg), expected)
                    self.assertEqual(self.size(warm, self.make_config('')), 32)

    def test_partitioned_tasks_cover_exact_selection(self):
        for cls in (SizePartitioner, NumWorkerPartitioner):
            for strategy in ('split', 'heuristic'):
                with self.subTest(partitioner=cls, strategy=strategy):
                    partitioner = self.make_partitioner(cls, strategy)
                    partitioner._dataset_size = {}
                    cfg = self.make_config('[4:20]')
                    original = copy.deepcopy(cfg)
                    tasks = partitioner.partition([
                        dict(models=[ConfigDict(abbr='model')], datasets=[cfg])
                    ],
                                                  work_dir=self.temp_dir.name,
                                                  out_dir=self.temp_dir.name)
                    indices = []
                    for task in tasks:
                        for part in task.datasets[0]:
                            indices.extend(
                                build_dataset_from_cfg(part).test['index'])
                    self.assertEqual(sorted(indices), list(range(4, 20)))
                    self.assertGreater(len(tasks), 1)
                    self.assertEqual(cfg, original)

    def test_force_rebuild_replaces_sliced_size(self):
        partitioner = self.make_partitioner(NumWorkerPartitioner,
                                            force_rebuild=True)
        with open(partitioner.dataset_size_path, 'w') as cache_file:
            json.dump({'cache-test': 16}, cache_file)
        self.assertEqual(partitioner.get_size(self.make_config('[4:20]')), 16)
        self.assertEqual(partitioner.dataset_size, {'cache-test': 32})

    def test_cost_factor_preserved_on_cache_miss(self):
        partitioner = self.make_partitioner(SizePartitioner)
        cfg = self.make_config('[4:20]')
        cfg.infer_cfg.prompt_template.template = {'A': 'a', 'B': 'b'}
        self.assertEqual(partitioner.get_cost(cfg, get_raw_factors=True),
                         (16, 2))
        self.assertEqual(partitioner.get_cost(cfg), 32)


if __name__ == '__main__':
    unittest.main()
