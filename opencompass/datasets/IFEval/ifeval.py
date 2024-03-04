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
        results = dict()
        for metric in ('strict', 'loose'):
            results[metric] = []
        for pred, refer in zip(predictions, references):
            input = InputExample(
                key=refer['key'],
                instruction_id_list=refer['instruction_id_list'],
                prompt=refer['prompt'],
                kwargs=refer['kwargs'])
            for kwarg in input.kwargs:
                for k in list(kwarg.keys()):
                    if kwarg[k] is None:
                        kwarg.pop(k, None)
            results['strict'].append(
                test_instruction_following_strict(input, pred))
            results['loose'].append(
                test_instruction_following_loose(input, pred))
        final_scores = dict()
        for metric in ('strict', 'loose'):
            prompt_total = 0
            prompt_correct = 0
            inst_total = 0
            inst_correct = 0

            for example in results[metric]:
                follow_instruction_list = example.follow_instruction_list
                instruction_id_list = example.instruction_id_list

                prompt_total += 1
                if all(follow_instruction_list):
                    prompt_correct += 1

                inst_total += len(instruction_id_list)
                inst_correct += sum(follow_instruction_list)
            prompt_score = f'Prompt-level-{metric}-accuracy'
            inst_score = f'Inst-level-{metric}-accuracy'
            final_scores[prompt_score] = prompt_correct / prompt_total * 100
            final_scores[inst_score] = inst_correct / inst_total * 100
        return final_scores
