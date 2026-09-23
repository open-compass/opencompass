"""Offline tests for the CivilComments loader and configs."""

import contextlib
import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = REPO_ROOT / 'opencompass' / 'datasets' / 'civilcomments.py'
CONFIG_DIR = (REPO_ROOT / 'opencompass' / 'configs' / 'datasets' /
              'civilcomments')


class _Registry:

    def register_module(self):

        def decorator(cls):
            return cls

        return decorator


class _FakeDataset:

    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]

    def remove_columns(self, columns):
        return _FakeDataset([{
            key: value
            for key, value in row.items() if key not in columns
        } for row in self.rows])

    def shuffle(self, seed):
        assert seed == 42
        return self

    def select(self, indices):
        return _FakeDataset(
            [self.rows[index] for index in indices if index < len(self.rows)])

    def map(self, function):
        return _FakeDataset([function(dict(row)) for row in self.rows])

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


class _DatasetDict(dict):
    pass


class AccEvaluator:
    pass


class AUCROCEvaluator:
    pass


class CLPInferencer:
    pass


class CivilCommentsDatasetStub:
    pass


class GenInferencer:
    pass


class PromptTemplate:
    pass


class ZeroRetriever:
    pass


def first_option_postprocess(text, options, cushion=True):
    del text, options, cushion


def _make_package(name, path=None):
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]
    return module


def _make_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class TestCivilCommentsDataset(unittest.TestCase):

    def test_load_maps_threshold_to_consistent_numeric_and_ab_labels(self):
        test_rows = [
            dict(text='safe', toxicity=0.4999),
            dict(text='boundary', toxicity=0.5),
            dict(text='toxic', toxicity=0.9),
        ]

        def fake_load_dataset(**kwargs):
            if kwargs['split'] == 'train':
                return _FakeDataset([dict(text='train', toxicity=0.1)])
            return _FakeDataset(test_rows)

        stubs = {
            'datasets':
            _make_module(
                'datasets',
                DatasetDict=_DatasetDict,
                load_dataset=fake_load_dataset,
            ),
            'opencompass':
            _make_package('opencompass'),
            'opencompass.datasets':
            _make_package('opencompass.datasets'),
            'opencompass.datasets.base':
            _make_module('opencompass.datasets.base', BaseDataset=object),
            'opencompass.registry':
            _make_module('opencompass.registry', LOAD_DATASET=_Registry()),
        }
        module_name = 'opencompass.datasets.civilcomments'

        with patch.dict(sys.modules, stubs):
            spec = importlib.util.spec_from_file_location(
                module_name, DATASET_FILE)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                dataset = module.CivilCommentsDataset.load(
                    path='civil_comments')
            finally:
                sys.modules.pop(module_name, None)

        self.assertEqual(
            [(row['label'], row['answer'], row['choices'])
             for row in dataset['test'].rows],
            [
                (0, 'A', ['no', 'yes']),
                (1, 'B', ['no', 'yes']),
                (1, 'B', ['no', 'yes']),
            ],
        )


class TestCivilCommentsConfigs(unittest.TestCase):

    def setUp(self):
        self.module_names = [
            'opencompass.configs.datasets.civilcomments.civilcomments_clp',
            'opencompass.configs.datasets.civilcomments.'
            'civilcomments_clp_a3c5fd',
            'opencompass.configs.datasets.civilcomments.civilcomments_gen',
            'opencompass.configs.datasets.civilcomments.'
            'civilcomments_gen_9d530e',
        ]
        for module_name in self.module_names:
            sys.modules.pop(module_name, None)

        stubs = {
            'mmengine':
            _make_package('mmengine'),
            'mmengine.config':
            _make_module('mmengine.config',
                         read_base=lambda: contextlib.nullcontext()),
            'opencompass':
            _make_package('opencompass', REPO_ROOT / 'opencompass'),
            'opencompass.configs':
            _make_package('opencompass.configs',
                          REPO_ROOT / 'opencompass' / 'configs'),
            'opencompass.configs.datasets':
            _make_package('opencompass.configs.datasets',
                          REPO_ROOT / 'opencompass' / 'configs' / 'datasets'),
            'opencompass.configs.datasets.civilcomments':
            _make_package('opencompass.configs.datasets.civilcomments',
                          CONFIG_DIR),
            'opencompass.datasets':
            _make_module(
                'opencompass.datasets',
                CivilCommentsDataset=CivilCommentsDatasetStub,
            ),
            'opencompass.openicl':
            _make_package('opencompass.openicl'),
            'opencompass.openicl.icl_evaluator':
            _make_module(
                'opencompass.openicl.icl_evaluator',
                AccEvaluator=AccEvaluator,
                AUCROCEvaluator=AUCROCEvaluator,
            ),
            'opencompass.openicl.icl_inferencer':
            _make_module(
                'opencompass.openicl.icl_inferencer',
                CLPInferencer=CLPInferencer,
                GenInferencer=GenInferencer,
            ),
            'opencompass.openicl.icl_prompt_template':
            _make_module('opencompass.openicl.icl_prompt_template',
                         PromptTemplate=PromptTemplate),
            'opencompass.openicl.icl_retriever':
            _make_module('opencompass.openicl.icl_retriever',
                         ZeroRetriever=ZeroRetriever),
            'opencompass.utils':
            _make_package('opencompass.utils'),
            'opencompass.utils.text_postprocessors':
            _make_module(
                'opencompass.utils.text_postprocessors',
                first_option_postprocess=first_option_postprocess,
            ),
        }
        self.module_patch = patch.dict(sys.modules, stubs)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.addCleanup(self._drop_modules)

    def _drop_modules(self):
        for module_name in self.module_names:
            sys.modules.pop(module_name, None)

    def test_gen_config_is_strict_ab_accuracy_config(self):
        module = importlib.import_module(
            'opencompass.configs.datasets.civilcomments.civilcomments_gen')
        dataset = module.civilcomments_datasets[0]
        prompt = dataset['infer_cfg']['prompt_template']['template']['round'][
            0]['prompt']
        postprocessor = dataset['eval_cfg']['pred_postprocessor']

        self.assertEqual(dataset['reader_cfg']['output_column'], 'answer')
        self.assertIs(dataset['infer_cfg']['inferencer']['type'],
                      GenInferencer)
        self.assertIs(dataset['eval_cfg']['evaluator']['type'], AccEvaluator)
        self.assertIn('A. No\nB. Yes', prompt)
        self.assertIn('exactly one letter, A or B', prompt)
        self.assertIs(postprocessor['type'], first_option_postprocess)
        self.assertEqual(postprocessor['options'], 'AB')
        self.assertFalse(postprocessor['cushion'])

    def test_clp_stable_config_remains_auc_based(self):
        module = importlib.import_module(
            'opencompass.configs.datasets.civilcomments.civilcomments_clp')
        dataset = module.civilcomments_datasets[0]

        self.assertEqual(dataset['reader_cfg']['output_column'], 'label')
        self.assertIs(dataset['infer_cfg']['inferencer']['type'],
                      CLPInferencer)
        self.assertIs(dataset['eval_cfg']['evaluator']['type'],
                      AUCROCEvaluator)

    def test_dataset_index_exposes_both_stable_configs(self):
        dataset_index = (REPO_ROOT / 'dataset-index.yml').read_text()

        self.assertIn(
            'opencompass/configs/datasets/civilcomments/civilcomments_clp.py',
            dataset_index,
        )
        self.assertIn(
            'opencompass/configs/datasets/civilcomments/civilcomments_gen.py',
            dataset_index,
        )


if __name__ == '__main__':
    unittest.main()
