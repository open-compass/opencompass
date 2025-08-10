import importlib
import json
import os.path as osp

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class InternSandboxDataset(BaseDataset):

    @staticmethod
    def load(path: str, sandbox: str, local_mode: bool = False):
        path = get_data_path(path, local_mode=local_mode)
        file_path = osp.join(path, f'{sandbox}.jsonl')
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                origin_data = json.loads(line)
                origin_data['ground_truth'] = json.dumps(
                    origin_data['ground_truth'])
                data.append(origin_data)
        return Dataset.from_list(data)


@ICL_EVALUATORS.register_module()
class InternSandboxEvaluator(BaseEvaluator):

    def __init__(self,
                 short_penalty: bool = False,
                 format_penalty: bool = False):
        super().__init__()
        self.short_penalty = short_penalty
        self.format_penalty = format_penalty

    def score(self, predictions, references, test_set):

        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }

        class_name = f"{test_set[0]['data_source']}Sandbox"

        details = []
        for pred, ref, ts in zip(predictions, references, test_set):
            ref = json.loads(ref)
            module = importlib.import_module('intern_sandbox')
            score = getattr(module, class_name).verify_score(
                pred,
                ref,
                short_penalty=self.short_penalty,
                format_penalty=self.format_penalty)
            try:
                extracted = getattr(module, class_name).extract_output(pred)
            except:  # noqa: E722
                extracted = None

            res = {
                'prompt': ts['prompt'],
                'score': score,
                'extracted_output': extracted,
                'ground_truth': ref,
                'output': pred,
            }
            details.append(res)

        avg_score = sum(r['score'] for r in details) / len(details)
        results = {'accuracy': avg_score, 'details': details}
        return results
