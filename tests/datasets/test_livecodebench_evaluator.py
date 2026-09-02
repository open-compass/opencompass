import json
import resource
import unittest
from unittest.mock import patch

from opencompass.datasets.livecodebench import evaluator, testing_util


class TestLiveCodeBenchMemoryLimit(unittest.TestCase):

    def test_run_test_passes_memory_limit_to_reliability_guard(self):
        sample = {
            'input_output':
            json.dumps({
                'inputs': ['1'],
                'outputs': ['1'],
                'fn_name': 'identity',
            })
        }

        with patch.object(testing_util,
                          'reliability_guard') as mock_reliability_guard:
            testing_util.run_test(sample,
                                  test='class Solution:\n    pass\n',
                                  timeout=1,
                                  memory_limit_bytes=123456)

        mock_reliability_guard.assert_called_once_with(
            maximum_memory_bytes=123456)

    def test_codegen_check_correctness_passes_memory_limit_to_worker(self):

        def fake_run_test(sample,
                          test=None,
                          debug=False,
                          timeout=6,
                          memory_limit_bytes=None):
            return [True], {'memory_limit_bytes': memory_limit_bytes}

        sample = {
            'input_output':
            json.dumps({
                'inputs': ['1'],
                'outputs': ['1'],
                'fn_name': 'identity',
            })
        }

        with patch.object(testing_util, 'run_test', fake_run_test):
            result, metadata = evaluator.codegen_check_correctness(
                sample,
                'unused generation',
                timeout=1,
                debug=False,
                memory_limit_bytes=123456)

        self.assertEqual(result, [True])
        self.assertEqual(metadata['memory_limit_bytes'], 123456)

    def test_reliability_guard_adds_memory_limit_to_current_vmsize(self):
        child_memory_limit = 256 * 1024 * 1024
        baseline_vmsize_bytes = 64 * 1024 * 1024 * 1024

        def fake_run_test(sample,
                          test=None,
                          debug=False,
                          timeout=6,
                          memory_limit_bytes=None):
            rlimit_data_before = resource.getrlimit(resource.RLIMIT_DATA)
            rlimit_stack_before = resource.getrlimit(resource.RLIMIT_STACK)
            testing_util.reliability_guard(
                maximum_memory_bytes=memory_limit_bytes)
            rlimit_data_unchanged = (resource.getrlimit(
                resource.RLIMIT_DATA) == rlimit_data_before)
            rlimit_stack_unchanged = (resource.getrlimit(
                resource.RLIMIT_STACK) == rlimit_stack_before)
            return [True], {
                'baseline_vmsize_bytes': baseline_vmsize_bytes,
                'rlimit_as': list(resource.getrlimit(resource.RLIMIT_AS)),
                'rlimit_data_unchanged': rlimit_data_unchanged,
                'rlimit_stack_unchanged': rlimit_stack_unchanged,
            }

        sample = {
            'input_output':
            json.dumps({
                'inputs': ['1'],
                'outputs': ['1'],
                'fn_name': 'identity',
            })
        }

        with patch.object(testing_util,
                          '_get_current_vmsize_bytes',
                          return_value=baseline_vmsize_bytes), patch.object(
                              testing_util, 'run_test', fake_run_test):
            result, metadata = evaluator.codegen_check_correctness(
                sample,
                'unused generation',
                timeout=1,
                debug=False,
                memory_limit_bytes=child_memory_limit)

        self.assertEqual(result, [True])
        effective_limit = (metadata['baseline_vmsize_bytes'] +
                           child_memory_limit)
        self.assertEqual(metadata['rlimit_as'],
                         [effective_limit, effective_limit])
        self.assertTrue(metadata['rlimit_data_unchanged'])
        self.assertTrue(metadata['rlimit_stack_unchanged'])

    def test_codegen_check_correctness_returns_metadata_when_worker_exits(
            self):

        def fake_run_test(sample,
                          test=None,
                          debug=False,
                          timeout=6,
                          memory_limit_bytes=None):
            raise SystemExit(1)

        sample = {
            'input_output':
            json.dumps({
                'inputs': ['1'],
                'outputs': ['1'],
                'fn_name': 'identity',
            })
        }

        with patch.object(testing_util, 'run_test', fake_run_test):
            result, metadata = evaluator.codegen_check_correctness(
                sample,
                'unused generation',
                timeout=1,
                debug=False,
                memory_limit_bytes=123456)

        self.assertEqual(result, [-1])
        self.assertEqual(metadata['error_message'],
                         'Global Timeout or Memory Limit Exceeded')


if __name__ == '__main__':
    unittest.main()
