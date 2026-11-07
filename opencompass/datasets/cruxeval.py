# flake8: noqa
# yapf: disable
import ast
import multiprocessing
from typing import Dict, List

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils.code_execution import (TYPE_AWARE_EQUAL_NAME,
                                              make_assertions_type_aware,
                                              type_aware_equal)

from .base import BaseDataset

CRUXEVAL_INPUT_FIELDS = ['code', 'input']


def _cruxeval_exec(code, test_case, queue):
    """Execute code in a subprocess to allow timeout."""
    try:
        exec_globals = {'__builtins__': __builtins__}
        exec(code, exec_globals)
        exec_globals[TYPE_AWARE_EQUAL_NAME] = type_aware_equal
        exec(test_case, exec_globals)
        queue.put(True)
    except Exception:
        queue.put(False)


def _make_type_aware_test_case(output: str, prediction: str) -> str:
    """Build an assertion without interpolating untrusted source around it."""
    output_expr = ast.parse(output, mode='eval').body
    prediction_expr = ast.parse(prediction, mode='eval').body
    assertion = ast.Module(
        body=[
            ast.Assert(
                test=ast.Compare(
                    left=output_expr,
                    ops=[ast.Eq()],
                    comparators=[prediction_expr],
                ),
                msg=None,
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(assertion)
    return make_assertions_type_aware(ast.unparse(assertion))


def _check_correctness(code: str,
                       output: str,
                       prediction: str,
                       timeout: int = 3) -> bool:
    """Check a type-aware equality assertion against the provided code.

    Follows the official CRUXEval evaluation logic.
    """
    try:
        test_case = _make_type_aware_test_case(output, prediction)
        ctx = multiprocessing.get_context('fork')
        queue = ctx.Queue()
        proc = ctx.Process(target=_cruxeval_exec,
                           args=(code, test_case, queue))
        proc.start()
        proc.join(timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return False
        return queue.get_nowait()
    except Exception:
        return False


def cruxeval_o_postprocess(text: str) -> str:
    """Extract the predicted output value from the model generation.

    The official Phind prompt expects the model to produce the value followed
    by ``# done``. We strip everything from ``# done`` onwards and take the
    first remaining line.
    """
    text = text.split('# done')[0]
    text = text.strip()
    text = text.split('\n')[0].strip()
    return text


@LOAD_DATASET.register_module()
class CRUXEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str = 'cruxeval-org/cruxeval'):
        from datasets import load_dataset
        dataset = load_dataset(path, split='test')
        return dataset


@ICL_EVALUATORS.register_module()
class CRUXEvalOEvaluator(BaseEvaluator):
    """Execution-based evaluator for CRUXEval-O (output prediction).

    For each sample the prediction is checked by running::

        {code}
        assert {output} == {prediction}

    The equality assertion is rewritten to require recursively matching
    concrete types, preventing predictions from forging equality through a
    custom ``__eq__`` implementation.

    A prediction is also rejected (anti-cheat) when it simply echoes
    ``f({input})`` instead of computing the real value, following the
    official implementation.
    """

    def __init__(self, timeout: int = 3) -> None:
        super().__init__()
        self.timeout = timeout

    def score(self, predictions, references, test_set):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        details = {}
        correct = 0
        for idx, prediction in enumerate(predictions):
            item = test_set[idx]
            code = item['code']
            inp = item['input']
            out = item['output']

            preds = prediction if isinstance(prediction, list) else [prediction]

            execution_results = []
            for g in preds:
                if not isinstance(g, str):
                    g = str(g)
                # anti-cheat: skip if model just echoes f(input)
                if f'f({inp})' in g:
                    continue
                execution_results.append(
                    _check_correctness(code, out, g, self.timeout))

            if True not in execution_results:
                execution_results = [False] * len(preds)

            is_correct = any(execution_results)
            correct += int(is_correct)

            details[str(idx)] = {
                'id': item.get('id', idx),
                'prediction': prediction,
                'reference': out,
                'is_correct': is_correct,
            }

        total = len(predictions)
        accuracy = correct / total * 100 if total > 0 else 0.0
        return {'CRUXEval-O': accuracy, 'details': details}
