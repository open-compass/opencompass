import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[2] / 'opencompass/runners/base.py'


def _load_base_runner():
    mmengine = types.ModuleType('mmengine')
    mmengine_config = types.ModuleType('mmengine.config')
    mmengine_config.Config = lambda task: SimpleNamespace(**task)
    mmengine_config.ConfigDict = dict
    mmengine.config = mmengine_config

    opencompass = types.ModuleType('opencompass')
    opencompass_utils = types.ModuleType('opencompass.utils')
    opencompass_utils.LarkReporter = object
    opencompass_utils.get_logger = lambda: SimpleNamespace(error=lambda *_: None)
    opencompass.utils = opencompass_utils

    stubs = {
        'mmengine': mmengine,
        'mmengine.config': mmengine_config,
        'opencompass': opencompass,
        'opencompass.utils': opencompass_utils,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            'opencompass_runner_base', MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.BaseRunner
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


BaseRunner = _load_base_runner()


class _FakeRunner(BaseRunner):

    def __init__(self, status):
        super().__init__(task={'type': 'fake'})
        self.status = status

    def launch(self, tasks):
        return self.status


class TestBaseRunner(unittest.TestCase):

    def test_successful_tasks_return_normally(self):
        runner = _FakeRunner([('task-a', 0), ('task-b', 0)])
        runner([])

    def test_failed_tasks_raise_runtime_error(self):
        runner = _FakeRunner([('task-a', 0), ('task-b', 1)])

        with self.assertRaisesRegex(RuntimeError, 'task-b'):
            runner([])


if __name__ == '__main__':
    unittest.main()
