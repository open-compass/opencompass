import sys
from types import ModuleType

from opencompass.datasets.gsm8k import GSM8KDataset


def test_modelscope_uses_main_config(monkeypatch):
    calls = {}
    modelscope = ModuleType('modelscope')

    class MsDataset:
        @staticmethod
        def load(**kwargs):
            calls.update(kwargs)
            return object()

    modelscope.MsDataset = MsDataset
    monkeypatch.setenv('DATASET_SOURCE', 'ModelScope')
    monkeypatch.setitem(sys.modules, 'modelscope', modelscope)

    GSM8KDataset.load('gsm8k')

    assert calls['subset_name'] == 'main'
