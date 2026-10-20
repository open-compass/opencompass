import re
from typing import List

from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import iter_jsonl


@LOAD_DATASET.register_module()
class InfiniteBenchmathcalcDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)

        dataset = list(iter_jsonl(path))

        raw_data = []
        for item in dataset:
            context = item['context']
            answer = item['answer']
            raw_data.append({'context': context, 'answer': answer})
        dataset = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module()
class InfiniteBenchmathcalcEvaluator(BaseEvaluator):

    def score(self, predictions: List, references: List) -> dict:
        score = 0.
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference = references[i]
            prediction_nums = []
            prediction_list = re.split('[^0-9]', prediction)
            for item in prediction_list:
                if item != '':
                    prediction_nums.append(int(item))

            for j in range(len(reference)):
                if j >= len(prediction_nums):
                    break

                if reference[j] == prediction_nums[j]:
                    score += 1
                else:
                    break

        score = score / len(predictions) * 100
        return {'score': score}
