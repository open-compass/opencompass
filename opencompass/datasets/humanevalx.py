from typing import Iterable, Dict, List
import os.path as osp
import gzip
import tempfile
import json
import re
from datasets import Dataset

from .base import BaseDataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.openicl.icl_evaluator import BaseEvaluator

@LOAD_DATASET.register_module()
class HumanevalXDataset(BaseDataset):

    @staticmethod
    def load(path, language='python', **kwargs):
        file_path = osp.join(path, f"humanevalx_{language}.jsonl.gz")
        dataset = HumanevalXDataset._stream_jsonl_all(file_path)
        return Dataset.from_list(dataset)
    
    @staticmethod
    def _stream_jsonl_all(filename: str) -> Iterable[Dict]:
        results = []
        if filename.endswith(".gz"):
            fp = gzip.open(open(filename, "rb"), "rt")
        else:
            fp = open(filename, "r")
        for line in fp:
            if any(not x.isspace() for x in line):
                results.append(json.loads(line))
        fp.close()

        return results

class HumanevalXEvaluator(BaseEvaluator):
    """Evaluator for human eval."""

    def __init__(self, k: List[int] = [1, 10, 100]) -> None:
        self.k = k
        super().__init__()

    def score(self, predictions, references):
        predictions = [{
            'task_id': f'HumanEval/{i}',
            'completion': predictions[i]
        } for i in range(len(predictions))]
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     out_dir = osp.join(tmp_dir, 'human_eval.json')
        #     self.write_jsonl(out_dir, predictions)
        #     score = self.eval(out_dir,
        #                       self.k,
        #                       n_workers=4,
        #                       timeout=3.0,
        #                       problem_file=self.HUMAN_EVAL)
        #     return {f'humaneval_{k}': score[k] * 100 for k in score}


def humanevalx_postprocess(text: str) -> str:
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

