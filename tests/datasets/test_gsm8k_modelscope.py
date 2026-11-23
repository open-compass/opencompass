import sys
from types import ModuleType

from opencompass.datasets.gsm8k import GSM8KDataset


def test_modelscope_uses_main_config(monkeypatch):
    calls = {}
    expected_dataset = object()
    modelscope = ModuleType('modelscope')

    class MsDataset:

        @staticmethod
        def load(**kwargs):
            calls.update(kwargs)
            return expected_dataset

    modelscope.MsDataset = MsDataset
    monkeypatch.setenv('DATASET_SOURCE', 'ModelScope')
    monkeypatch.setitem(sys.modules, 'modelscope', modelscope)

    dataset = GSM8KDataset.load('opencompass/gsm8k')

    assert dataset is expected_dataset
    assert calls == {
        'dataset_name': 'opencompass/gsm8k',
        'subset_name': 'main',
    }
