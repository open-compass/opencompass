# flake8: noqa
# yapf: disable
import multiprocessing
from typing import Dict, List

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

CRUXEVAL_INPUT_FIELDS = ['code', 'input']


def _cruxeval_exec(code, queue):
    """Execute code in a subprocess to allow timeout."""
    try:
        exec(code, {'__builtins__': __builtins__})
        queue.put(True)
    except Exception:
        queue.put(False)


def _check_correctness(code: str, timeout: int = 3) -> bool:
    """Check if ``assert {output} == {prediction}`` holds given the code.

    Follows the official CRUXEval evaluation logic.
    """
    try:
        ctx = multiprocessing.get_context('fork')
        queue = ctx.Queue()
        proc = ctx.Process(target=_cruxeval_exec, args=(code, queue))
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
                code_to_execute = f'{code}\nassert {out} == {g}'
                execution_results.append(
                    _check_correctness(code_to_execute, self.timeout))

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
