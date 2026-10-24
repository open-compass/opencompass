import json
import multiprocessing
import resource
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opencompass.datasets.livecodebench import evaluator, testing_util


def _run_non_linux_reliability_guard(result_queue):
    import warnings

    resource_module = resource
    setrlimit_calls = []
    original_setrlimit = resource_module.setrlimit
    original_uname = testing_util.platform.uname
    original_sys_modules_resource = sys.modules.get('resource')
    resource_module.setrlimit = lambda *args: setrlimit_calls.append(args)
    testing_util.platform.uname = lambda: SimpleNamespace(system='Darwin')
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            testing_util.reliability_guard(maximum_memory_bytes=123456)
        sys.modules['resource'] = resource_module
        result_queue.put({
            'warning_count':
            len(caught),
            'warning_message':
            str(caught[0].message) if caught else None,
            'setrlimit_calls':
            setrlimit_calls,
        })
    finally:
        sys.modules['resource'] = original_sys_modules_resource
        resource_module.setrlimit = original_setrlimit
        testing_util.platform.uname = original_uname


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

    def test_reliability_guard_skips_memory_limit_on_non_linux(self):
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_run_non_linux_reliability_guard, args=(result_queue, ))
        process.start()
        process.join(timeout=5)

        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, 0)
        result = result_queue.get(timeout=1)
        process.close()

        self.assertEqual(result['warning_count'], 1)
        self.assertIn('only supported on Linux', result['warning_message'])
        self.assertEqual(result['setrlimit_calls'], [])

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


class TestLCBExecutorIOGate(unittest.TestCase):

    def test_stdin_buffer_read_and_multiline_readline(self):
        mock = testing_util.MockStdinWithBuffer('first\nsecond\nthird')
        self.assertEqual(mock.buffer.readline(), b'first\n')
        self.assertEqual(mock.buffer.readline(), b'second\n')
        self.assertEqual(mock.buffer.readline(), b'third')
        self.assertEqual(mock.buffer.readline(), b'')
        self.assertEqual(
            testing_util.MockStdinWithBuffer('a\nb\n').buffer.read(),
            b'a\nb\n')

    def test_stdout_buffer_write_is_captured(self):
        import sys

        with testing_util.Capturing() as output:
            sys.stdout.buffer.write(b'hello\n')
            sys.stdout.buffer.write(b'world\n')
        self.assertEqual(output[0], 'hello\nworld\n')

    def test_call_method_echoes_stdin_buffer_to_stdout_buffer(self):

        def echo():
            import sys
            sys.stdout.buffer.write(sys.stdin.buffer.read())

        with testing_util.Capturing() as output:
            testing_util.call_method(echo, 'alpha\nbeta\n')
        self.assertIn('alpha', output[0])
        self.assertIn('beta', output[0])

    def test_check_lcb_io_executor_passes_current_copy(self):
        from opencompass.datasets.livecodebench.executor_gate import (
            LCB_IO_GATE_VERSION,
            check_lcb_io_executor,
        )

        provenance = check_lcb_io_executor()
        self.assertEqual(provenance['io_gate_version'], LCB_IO_GATE_VERSION)
        self.assertEqual(provenance['io_self_test'], 'passed')
        self.assertEqual(len(provenance['testing_util_sha256']), 64)

    def test_gate_rejects_call_method_without_buffer_mock(self):
        from opencompass.datasets.livecodebench.executor_gate import (
            LCBExecutorGateError,
            check_lcb_io_executor,
        )
        from opencompass.datasets.livecodebench import testing_util as lcb_testing_util

        from io import StringIO
        from unittest.mock import patch as mock_patch

        def fake_call_method(method, inputs):
            with mock_patch('sys.stdin', StringIO(inputs)):
                return method()

        with patch.object(lcb_testing_util, 'call_method', fake_call_method):
            with self.assertRaises(LCBExecutorGateError) as ctx:
                check_lcb_io_executor()
        self.assertIn('Refusing to emit Coding scores', str(ctx.exception))

    def test_stringio_buffer_failures_block_official_score(self):
        from opencompass.datasets.livecodebench.executor_gate import (
            LCBExecutorGateError,
            assert_lcb_infra_exceptions_within_budget,
        )

        error = (
            'AttributeError("\'_io.StringIO\' object has no attribute '
            '\'buffer\'")')
        metadata = [
            [json.dumps({'error': error, 'error_message': 'Runtime Error'})]
            for _ in range(3)
        ]
        with self.assertRaises(LCBExecutorGateError) as ctx:
            assert_lcb_infra_exceptions_within_budget(metadata)
        self.assertIn('Refusing to emit Coding scores', str(ctx.exception))

    def test_infra_failures_within_budget_do_not_block(self):
        from opencompass.datasets.livecodebench.executor_gate import (
            assert_lcb_infra_exceptions_within_budget,
        )

        error = (
            'AttributeError("\'_io.StringIO\' object has no attribute '
            '\'buffer\'")')
        metadata = [[json.dumps({'error_message': 'Wrong Answer'})]
                    for _ in range(100)]
        metadata[0] = [
            json.dumps({
                'error': error,
                'error_message': 'Runtime Error'
            })
        ]
        stats = assert_lcb_infra_exceptions_within_budget(metadata)
        self.assertEqual(stats['stringio_buffer_failures'], 1)

    def test_evaluator_score_returns_error_when_gate_fails(self):
        from opencompass.datasets.livecodebench.evaluator import (
            LCBCodeGenerationEvaluator,
        )
        from opencompass.datasets.livecodebench.executor_gate import (
            LCBExecutorGateError,
        )

        fake_eval = LCBCodeGenerationEvaluator.__new__(
            LCBCodeGenerationEvaluator)
        with patch.object(evaluator, 'check_lcb_io_executor',
                          side_effect=LCBExecutorGateError('rolled back')):
            result = fake_eval.score(['print(1)'], ['q1'])
        self.assertEqual(result, {'error': 'rolled back'})
        self.assertNotIn('pass@1', result)


if __name__ == '__main__':
    unittest.main()
