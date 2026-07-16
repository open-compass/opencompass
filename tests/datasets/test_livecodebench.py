import unittest
from unittest.mock import patch

from opencompass.datasets.livecodebench import evaluator


class TestLiveCodeBenchEvaluator(unittest.TestCase):

    def test_codegen_check_returns_failure_when_worker_exits(self):

        class FakeManager:

            def list(self):
                return []

        class FakeProcess:

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

            def join(self, timeout):
                pass

            def is_alive(self):
                return False

        sample = {'input_output': '{"inputs": [""], "outputs": [""]}'}
        with patch.object(evaluator.multiprocessing, 'Manager', FakeManager), \
                patch.object(evaluator.multiprocessing, 'Process', FakeProcess):
            result, metadata = evaluator.codegen_check_correctness(
                sample, '', timeout=1, debug=False)

        self.assertEqual(result, [-1])
        self.assertEqual(
            metadata,
            {
                'error_code': -4,
                'error_message': 'Runtime Error',
            },
        )

    def test_code_execution_checks_prediction_containing_input(self):
        result = evaluator.evaluate_score((['1'], ('x = 1', '1', '1')))

        self.assertEqual(result, [True])
