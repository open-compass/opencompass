import json

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset
from .evaluation_main import (InputExample, test_instruction_following_loose,
                              test_instruction_following_strict)


@LOAD_DATASET.register_module()
class IFEvalDataset(BaseDataset):

    @staticmethod
    def load(path):
        datasets = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                tmp = json.loads(line.strip())
                dataset = dict(prompt=tmp['prompt'], reference=tmp)
                datasets.append(dataset)
        return Dataset.from_list(datasets)


class IFEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        results = []
        for pred, refer in zip(predictions, references):
            print(refer)
            input = InputExample(
                key=refer['key'],
                instruction_id_list=refer['instruction_id_list'],
                prompt=refer['prompt'],
                kwargs=refer['kwargs'])
            for kwarg in input.kwargs:
                for k in list(kwarg.keys()):
                    if kwarg[k] is None:
                        kwarg.pop(k, None)
            result = dict(
                strict=test_instruction_following_strict(input, pred),
                loose=test_instruction_following_loose(input, pred),
            )
            results.append(result)
        strict = sum(
            [result['strict'].follow_all_instructions
             for result in results]) / len(results)
        loose = sum(
            [result['loose'].follow_all_instructions
             for result in results]) / len(results)
        return dict(strict_acc=strict, loose_acc=loose)
