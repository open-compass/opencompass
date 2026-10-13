"""Tests for SIQA dataset configuration defaults."""

import contextlib
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
SIQA_CONFIG_DIR = REPO_ROOT / 'opencompass' / 'configs' / 'datasets' / 'siqa'


class AccEvaluator:
    pass


class GenInferencer:
    pass


class PromptTemplate:
    pass


class SiqaDatasetV3:
    pass


class ZeroRetriever:
    pass


def first_option_postprocess(text, options, cushion=True):
    pass


def _make_package(name, path=None):
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]
    return module


def _make_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _make_stubs():
    return {
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
        'opencompass.configs.datasets.siqa':
        _make_package('opencompass.configs.datasets.siqa', SIQA_CONFIG_DIR),
        'opencompass.openicl':
        _make_package('opencompass.openicl'),
        'opencompass.openicl.icl_prompt_template':
        _make_module('opencompass.openicl.icl_prompt_template',
                     PromptTemplate=PromptTemplate),
        'opencompass.openicl.icl_retriever':
        _make_module('opencompass.openicl.icl_retriever',
                     ZeroRetriever=ZeroRetriever),
        'opencompass.openicl.icl_inferencer':
        _make_module('opencompass.openicl.icl_inferencer',
                     GenInferencer=GenInferencer),
        'opencompass.openicl.icl_evaluator':
        _make_module('opencompass.openicl.icl_evaluator',
                     AccEvaluator=AccEvaluator),
        'opencompass.datasets':
        _make_module('opencompass.datasets', SiqaDatasetV3=SiqaDatasetV3),
        'opencompass.utils':
        _make_package('opencompass.utils'),
        'opencompass.utils.text_postprocessors':
        _make_module('opencompass.utils.text_postprocessors',
                     first_option_postprocess=first_option_postprocess),
    }


def _drop_modules(module_names):
    for module_name in module_names:
        sys.modules.pop(module_name, None)


class TestSIQAConfig(unittest.TestCase):

    def setUp(self):
        module_names = [
            'opencompass.configs.datasets.siqa.siqa_gen',
            'opencompass.configs.datasets.siqa.siqa_gen_18632c',
        ]
        _drop_modules(module_names)

        self.module_patch = patch.dict(sys.modules, _make_stubs())
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.addCleanup(_drop_modules, module_names)

    def test_default_siqa_gen_uses_strict_accuracy_config(self):
        module = importlib.import_module(
            'opencompass.configs.datasets.siqa.siqa_gen')

        dataset = module.siqa_datasets[0]
        eval_cfg = dataset['eval_cfg']

        self.assertIs(dataset['type'], SiqaDatasetV3)
        self.assertIs(eval_cfg['evaluator']['type'], AccEvaluator)
        self.assertIs(eval_cfg['pred_postprocessor']['type'],
                      first_option_postprocess)
        self.assertEqual(eval_cfg['pred_postprocessor']['options'], 'ABC')
        self.assertFalse(eval_cfg['pred_postprocessor']['cushion'])


if __name__ == '__main__':
    unittest.main()
