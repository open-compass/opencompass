"""Regression tests for GPQA simple-eval post-processing."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GPQA_MODULE_PATH = ROOT / 'opencompass' / 'datasets' / 'gpqa.py'


class DummyRegistry:

    def register_module(self, *args, **kwargs):
        def decorator(obj):
            return obj

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return decorator


def load_gpqa_postprocess():
    """Load the GPQA module with lightweight stubs for optional deps."""

    opencompass_pkg = types.ModuleType('opencompass')
    opencompass_pkg.__path__ = [str(ROOT / 'opencompass')]

    datasets_pkg = types.ModuleType('opencompass.datasets')
    datasets_pkg.__path__ = [str(ROOT / 'opencompass' / 'datasets')]

    base_mod = types.ModuleType('opencompass.datasets.base')

    class BaseDataset:  # noqa: D401
        pass

    base_mod.BaseDataset = BaseDataset

    registry_mod = types.ModuleType('opencompass.registry')
    registry_mod.LOAD_DATASET = DummyRegistry()
    registry_mod.TEXT_POSTPROCESSORS = DummyRegistry()

    openicl_mod = types.ModuleType('opencompass.openicl')

    class BaseEvaluator:  # noqa: D401
        pass

    openicl_mod.BaseEvaluator = BaseEvaluator

    utils_mod = types.ModuleType('opencompass.utils')
    utils_mod.get_data_path = lambda path, local_mode=True: path

    datasets_mod = types.ModuleType('datasets')

    class Dataset:  # noqa: D401
        @classmethod
        def from_list(cls, data):
            return data

    datasets_mod.Dataset = Dataset

    sys.modules['opencompass'] = opencompass_pkg
    sys.modules['opencompass.datasets'] = datasets_pkg
    sys.modules['opencompass.datasets.base'] = base_mod
    sys.modules['opencompass.registry'] = registry_mod
    sys.modules['opencompass.openicl'] = openicl_mod
    sys.modules['opencompass.utils'] = utils_mod
    sys.modules['datasets'] = datasets_mod

    spec = importlib.util.spec_from_file_location('opencompass.datasets.gpqa',
                                                  GPQA_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GPQA_Simple_Eval_postprocess


GPQA_Simple_Eval_postprocess = load_gpqa_postprocess()


class TestGPQASimpleEvalPostprocess(unittest.TestCase):

    def test_returns_the_last_answer_match(self):
        text = (
            'Reasoning step 1.\n'
            'ANSWER: C\n'
            'Self-correction after review.\n'
            'ANSWER: B\n')

        self.assertEqual(GPQA_Simple_Eval_postprocess(text), 'B')

    def test_accepts_case_and_spacing_variants(self):
        text = 'analysis complete.\nAnSwEr   :   D'

        self.assertEqual(GPQA_Simple_Eval_postprocess(text), 'D')

    def test_returns_none_when_no_answer_match_exists(self):
        text = 'Reasoning only, no final answer token is present.'

        self.assertIsNone(GPQA_Simple_Eval_postprocess(text))


if __name__ == '__main__':
    unittest.main()
