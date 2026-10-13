# flake8: noqa
import json
import os.path as osp

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.datasets.phybench.box_extract import extract_boxed_latex
from opencompass.datasets.phybench.EED import EED
from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class PhyBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path)
        file_path = osp.join(path, 'PHYBench-fullques_v1.json')

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        inputs = [item['content'] for item in raw_data]
        targets = [item['answer'] for item in raw_data]

        return Dataset.from_dict({'input': inputs, 'target': targets})


@ICL_EVALUATORS.register_module()
class MathEEDEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        scores = []
        details = []
        for pred, ref in zip(predictions, references):
            pred_clean = extract_boxed_latex(pred)
            score, _, _, _ = EED(ref, pred_clean)
            scores.append(score)
            details.append({
                'pred': pred,
                'ref': ref,
                'pred_clean': pred_clean,
                'score': score
            })
        return {'accuracy': sum(scores) / len(scores), 'details': details}
