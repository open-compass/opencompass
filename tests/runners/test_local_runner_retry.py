"""Unit tests for LocalRunner retry (G5, PR-5).

subprocess.run is mocked, so no real commands are executed. Locks the
retry semantics aligned with SlurmRunner: a failed task is relaunched
until it succeeds or the retry budget is exhausted.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from mmengine.config import ConfigDict

from opencompass.runners.local import LocalRunner


class FakeCfg:

    def dump(self, path):
        with open(path, 'w') as f:
            f.write('# fake task cfg\n')


class FakeTask:

    def __init__(self, exit_codes):
        self.exit_codes = list(exit_codes)
        self.calls = 0
        self.name = 'fake_task'
        self.num_gpus = 0
        self.cfg = FakeCfg()

    def get_command(self, cfg_path, template):
        return 'echo fake-task'

    def get_log_path(self, file_extension='out'):
        return os.path.join(self.log_dir, f'{self.name}.{file_extension}')


def _run_result(code):
    result = unittest.mock.Mock()
    result.returncode = code
    return result


def _make_runner(**kwargs):
    runner = LocalRunner(
        task=ConfigDict(dict(type='OpenICLInferTask')),
        max_num_workers=1,
        tmp_dir=tempfile.mkdtemp(),
        **kwargs)
    return runner


def _launch_with_codes(runner, task, codes):
    """Patch subprocess.run to return the given exit codes in order."""
    with patch('opencompass.runners.local.subprocess.run') as mock_run:
        mock_run.side_effect = [_run_result(c) for c in codes]
        name, code = runner._launch(task, [], 0)
    return name, code, mock_run.call_count


class TestLocalRunnerRetry(unittest.TestCase):

    def setUp(self):
        self.task = FakeTask([])
        self.task.log_dir = tempfile.mkdtemp()

    def test_default_no_retry_preserves_behavior(self):
        """BC lock: default retry budget is zero, a failed task runs once."""
        runner = _make_runner()

        name, code, calls = _launch_with_codes(runner, self.task, [1])

        self.assertEqual(code, 1)
        self.assertEqual(calls, 1)

    def test_retry_until_success(self):
        runner = _make_runner(retry=2)

        name, code, calls = _launch_with_codes(runner, self.task, [1, 1, 0])

        self.assertEqual(code, 0)
        self.assertEqual(calls, 3)

    def test_retry_budget_exhausted(self):
        runner = _make_runner(retry=2)

        name, code, calls = _launch_with_codes(runner, self.task, [1, 1, 1])

        self.assertEqual(code, 1)
        self.assertEqual(calls, 3)

    def test_success_stops_retrying(self):
        runner = _make_runner(retry=3)

        name, code, calls = _launch_with_codes(runner, self.task, [0, 0, 0, 0])

        self.assertEqual(code, 0)
        self.assertEqual(calls, 1)


if __name__ == '__main__':
    unittest.main()
