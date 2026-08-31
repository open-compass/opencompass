import json
import os
import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from opencompass.datasets.humaneval import (HumanEvalEvaluator,
                                            HumanEvalPlusEvaluator,
                                            humaneval_postprocess_v2,
                                            humaneval_postprocess_v3)


class TestHumanevalPostprocess(unittest.TestCase):

    def test_v2_returns_plain_text_without_code_fence(self):
        raw = '    return x - int(x)\n'

        self.assertEqual(humaneval_postprocess_v2(raw),
                         'return x - int(x)\n')

    def test_v2_extracts_python_code_block(self):
        raw = '\n'.join([
            'Here is the implementation:',
            '```python',
            '    return x - int(x)',
            '```',
            'This solves the task.',
        ])

        self.assertEqual(humaneval_postprocess_v2(raw),
                         'return x - int(x)\n')

    def test_v2_extracts_bare_code_block(self):
        raw = '\n'.join([
            '```',
            '    return x - int(x)',
            '```',
        ])

        self.assertEqual(humaneval_postprocess_v2(raw),
                         'return x - int(x)\n')

    def test_v2_uses_first_code_block(self):
        raw = '\n'.join([
            '```python',
            '    return "first"',
            '```',
            '```python',
            '    return "second"',
            '```',
        ])

        self.assertEqual(humaneval_postprocess_v2(raw),
                         'return "first"\n')

    def test_v3_uses_last_code_block(self):
        raw = '\n'.join([
            '```python',
            '    return "first"',
            '```',
            '```python',
            '    return "second"',
            '```',
        ])

        self.assertEqual(humaneval_postprocess_v3(raw),
                         'return "second"\n')

    def test_v3_matches_v2_for_single_code_block(self):
        raw = '\n'.join([
            '```python',
            '    value = x - int(x)',
            '    return value',
            '```',
        ])

        expected = 'value = x - int(x)\n    return value\n'
        self.assertEqual(humaneval_postprocess_v2(raw), expected)
        self.assertEqual(humaneval_postprocess_v3(raw), expected)


class TestHumanEvalEvaluatorDetails(unittest.TestCase):

    def test_score_returns_details_from_mocked_evaluation_results(self):
        human_eval = ModuleType('human_eval')
        human_eval_data = ModuleType('human_eval.data')
        human_eval_evaluation = ModuleType('human_eval.evaluation')
        write_jsonl = Mock()

        def fake_evaluate(sample_file, *args, **kwargs):
            result_file = sample_file + '_results.jsonl'
            rows = [
                {
                    'task_id': 'HumanEval/0',
                    'completion': 'return 1',
                    'passed': True,
                    'result': 'passed',
                },
                {
                    'task_id': 'HumanEval/1',
                    'completion': 'return 0',
                    'passed': False,
                    'result': 'failed: assertion',
                },
            ]
            with open(result_file, 'w', encoding='utf-8') as result:
                for row in rows:
                    result.write(json.dumps(row) + '\n')
            return {'pass@1': 0.5}

        evaluate = Mock(side_effect=fake_evaluate)
        human_eval_data.HUMAN_EVAL = 'mock_problem_file.jsonl'
        human_eval_data.write_jsonl = write_jsonl
        human_eval_evaluation.evaluate_functional_correctness = evaluate

        modules = {
            'human_eval': human_eval,
            'human_eval.data': human_eval_data,
            'human_eval.evaluation': human_eval_evaluation,
        }
        with patch.dict(sys.modules, modules):
            result = HumanEvalEvaluator(k=[1]).score(
                predictions=['return 1', 'return 0'],
                references=['HumanEval/0', 'HumanEval/1'],
                test_set=[{
                    'prompt': 'def one():'
                }, {
                    'prompt': 'def two():'
                }],
            )

        self.assertEqual(result['humaneval_pass@1'], 50.0)
        self.assertEqual(
            result['details'], {
                '0': {
                    'task_id': 'HumanEval/0',
                    'completion': 'return 1',
                    'passed': True,
                    'result': 'passed',
                    'is_correct': True,
                    'prompt': 'def one():',
                },
                '1': {
                    'task_id': 'HumanEval/1',
                    'completion': 'return 0',
                    'passed': False,
                    'result': 'failed: assertion',
                    'is_correct': False,
                    'prompt': 'def two():',
                },
            })
        sample_file, samples = write_jsonl.call_args.args
        self.assertEqual(samples, [{
            'task_id': 'HumanEval/0',
            'completion': 'return 1'
        }, {
            'task_id': 'HumanEval/1',
            'completion': 'return 0'
        }])
        evaluate.assert_called_once_with(sample_file, [1],
                                         n_workers=4,
                                         timeout=3.0,
                                         problem_file='mock_problem_file.jsonl')
        self.assertFalse(os.path.exists(os.path.dirname(sample_file)))


class TestHumanEvalPlusEvaluatorDetails(unittest.TestCase):

    def test_score_returns_details_from_mocked_evaluation_results(self):
        evalplus = ModuleType('evalplus')
        evalplus_data = ModuleType('evalplus.data')
        evalplus_evaluate = ModuleType('evalplus.evaluate')
        write_jsonl = Mock()

        def fake_evaluate(flags):
            result_file = '{}_eval_results.json'.format(flags.samples[:-6])
            eval_results = {
                'eval': {
                    'HumanEval/0': [{
                        'base_status': 'pass',
                        'plus_status': 'pass',
                    }],
                    'HumanEval/1': [{
                        'base_status': 'pass',
                        'plus_status': 'fail',
                    }],
                }
            }
            with open(result_file, 'w', encoding='utf-8') as result:
                json.dump(eval_results, result)

        evaluate = Mock(side_effect=fake_evaluate)
        evalplus_data.write_jsonl = write_jsonl
        evalplus_evaluate.evaluate = evaluate

        modules = {
            'evalplus': evalplus,
            'evalplus.data': evalplus_data,
            'evalplus.evaluate': evalplus_evaluate,
        }
        with patch.dict(sys.modules, modules):
            result = HumanEvalPlusEvaluator(k=[1]).score(
                predictions=['    return 1', '    return 0'],
                references=['HumanEval/0', 'HumanEval/1'],
                test_set=[{
                    'prompt': 'def one():\n'
                }, {
                    'prompt': 'def two():\n'
                }],
            )

        self.assertEqual(result['humaneval_plus_pass@1'], 50.0)
        self.assertEqual(
            result['details'], {
                '0': {
                    'prompt': 'def one():\n',
                    'prediction': '    return 1',
                    'reference': 'HumanEval/0',
                    'base_result': 'pass',
                    'plus_result': 'pass',
                    'is_correct': True,
                },
                '1': {
                    'prompt': 'def two():\n',
                    'prediction': '    return 0',
                    'reference': 'HumanEval/1',
                    'base_result': 'pass',
                    'plus_result': 'fail',
                    'is_correct': False,
                },
            })
        sample_file, samples = write_jsonl.call_args.args
        self.assertEqual(samples, [{
            'task_id': 'HumanEval/0',
            'solution': 'def one():\n    return 1'
        }, {
            'task_id': 'HumanEval/1',
            'solution': 'def two():\n    return 0'
        }])
        flags = evaluate.call_args.args[0]
        self.assertEqual(flags.dataset, 'humaneval')
        self.assertEqual(flags.samples, sample_file)
        self.assertFalse(os.path.exists(os.path.dirname(sample_file)))


if __name__ == '__main__':
    unittest.main()
