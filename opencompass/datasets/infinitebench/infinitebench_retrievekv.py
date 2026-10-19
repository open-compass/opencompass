from typing import List

from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import iter_jsonl


@LOAD_DATASET.register_module()
class InfiniteBenchretrievekvDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)

        dataset = list(iter_jsonl(path))

        raw_data = []
        for item in dataset:
            context = item['context']
            input = item['input']
            answer = item['answer']
            raw_data.append({
                'context': context,
                'input': input,
                'answer': answer
            })
        dataset = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module()
class InfiniteBenchretrievekvEvaluator(BaseEvaluator):

    def score(self, predictions: List, references: List) -> dict:
        score = 0.
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference = references[i]
            for c in ['\n', ':', '\"', '\'', '.', ',', '?', '!', '{', '}']:
                prediction = prediction.replace(c, ' ')
            words = prediction.split()
            if reference in words:
                score += 1

        score = score / len(predictions) * 100
        return {'score': score}
