import os.path as osp
import tempfile
from typing import List

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, TEXT_POSTPROCESSORS


@ICL_EVALUATORS.register_module()
class HumanEvaluator(BaseEvaluator):
    """Evaluator for human eval."""

    def __init__(self, k: List[int] = [1, 10, 100]) -> None:
        try:
            from human_eval.data import HUMAN_EVAL, write_jsonl
            from human_eval.evaluation import evaluate_functional_correctness
            self.write_jsonl = write_jsonl
            self.HUMAN_EVAL = HUMAN_EVAL
            self.eval = evaluate_functional_correctness
        except ImportError:
            raise ImportError('Please install human_eval following'
                              'https://github.com/openai/human-eval/tree/'
                              'master#installation first.')
        self.k = k
        super().__init__()

    def score(self, predictions, references):

        predictions = [{
            'task_id': f'HumanEval/{i}',
            'completion': predictions[i]
        } for i in range(len(predictions))]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = osp.join(tmp_dir, 'human_eval.json')
            self.write_jsonl(out_dir, predictions)
            score = self.eval(out_dir,
                              self.k,
                              n_workers=4,
                              timeout=3.0,
                              problem_file=self.HUMAN_EVAL)
            return {f'humaneval_{k}': score[k] * 100 for k in score}


@TEXT_POSTPROCESSORS.register_module('humaneval')
def humaneval_postprocess(text: str) -> str:
    text = text.split('\n\n')[0]
    if '```' in text:
        text = text.split('```')[1]
    if text.strip().startswith('def'):
        text = '\n'.join(text.split('\n')[1:])
    if not text.startswith('    '):
        if text.startswith(' '):
            text = '    ' + text.lstrip()
        else:
            text = '\n'.join(['    ' + line for line in text.split('\n')])
    return text
