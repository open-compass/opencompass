import os.path as osp
import re
import tempfile
from typing import List

from opencompass.openicl.icl_evaluator import BaseEvaluator


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


def humaneval_postprocess(text: str) -> str:
    if '```' in text:
        blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split('```')[1]  # fall back to default strategy
        else:
            text = blocks[0]  # fetch the first code block
            if not text.startswith('\n'):  # in case starting with ```python
                text = text[max(text.find('\n') + 1, 0):]
    if text.strip().startswith('from') or text.strip().startswith('import'):
        def_idx = text.find('def')
        if def_idx != -1:
            text = text[max(text.find('\n', def_idx) + 1, 0):]
    text = text.split('\n\n')[0]
    if text.strip().startswith('def'):
        text = '\n'.join(text.split('\n')[1:])
    if not text.startswith('    '):
        if text.startswith(' '):
            text = '    ' + text.lstrip()
        else:
            text = '\n'.join(['    ' + line for line in text.split('\n')])
    return text


def humaneval_gpt_postprocess(text: str) -> str:
    """Better answer postprocessor for better instruction-aligned models like
    GPT."""
    if '```' in text:
        blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split('```')[1]  # fall back to default strategy
        else:
            text = blocks[0]  # fetch the first code block
            if not text.startswith('\n'):  # in case starting with ```python
                text = text[max(text.find('\n') + 1, 0):]
    if text.strip().startswith('from') or text.strip().startswith('import'):
        def_idx = text.find('def')
        if def_idx != -1:
            text = text[max(text.find('\n', def_idx) + 1, 0):]
    text = text.split('\n\n\n')[0]
    if text.strip().startswith('def'):
        text = '\n'.join(text.split('\n')[1:])
    if not text.startswith('    '):
        if text.startswith(' '):
            text = '    ' + text.lstrip()
        else:
            text = '\n'.join(['    ' + line for line in text.split('\n')])
    return text
