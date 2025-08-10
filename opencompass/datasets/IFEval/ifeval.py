import json

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .evaluation_main import (InputExample, test_instruction_following_loose,
                              test_instruction_following_strict)


@LOAD_DATASET.register_module()
class IFEvalDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        datasets = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                tmp = json.loads(line.strip())
                dataset = dict(prompt=tmp['prompt'], reference=tmp)
                datasets.append(dataset)
        return Dataset.from_list(datasets)


class IFEvaluator(BaseEvaluator):

    def score(self, predictions, references, origin_prompt):
        prompt_strict_correct, prompt_strict_total = 0, 0
        inst_strict_correct, inst_strict_total = 0, 0
        prompt_loose_correct, prompt_loose_total = 0, 0
        inst_loose_correct, inst_loose_total = 0, 0
        details = {}
        for index, (pred, refer) in enumerate(zip(predictions, references)):
            input = InputExample(
                key=refer['key'],
                instruction_id_list=refer['instruction_id_list'],
                prompt=refer['prompt'],
                kwargs=refer['kwargs'])
            for kwarg in input.kwargs:
                for k in list(kwarg.keys()):
                    if kwarg[k] is None:
                        kwarg.pop(k, None)

            # strict
            example = test_instruction_following_strict(input, pred)
            follow_instruction_list = example.follow_instruction_list
            instruction_id_list = example.instruction_id_list
            prompt_strict_total += 1
            is_strict_correct = all(follow_instruction_list)
            prompt_strict_correct += is_strict_correct
            inst_strict_total += len(instruction_id_list)
            inst_strict_correct += sum(follow_instruction_list)

            # loose
            example = test_instruction_following_loose(input, pred)
            follow_instruction_list = example.follow_instruction_list
            instruction_id_list = example.instruction_id_list
            prompt_loose_total += 1
            is_loose_correct = all(follow_instruction_list)
            prompt_loose_correct += is_loose_correct
            inst_loose_total += len(instruction_id_list)
            inst_loose_correct += sum(follow_instruction_list)

            if is_strict_correct:
                grade = 'strict'
            elif is_loose_correct:
                grade = 'loose'
            else:
                grade = 'none'

            details[str(index)] = {
                'prompt': origin_prompt[index],
                'pred': pred,
                'refer': refer,
                'is_strict_correct': is_strict_correct,
                'is_loose_correct': is_loose_correct,
                'is_correct': is_strict_correct,
                'grade': grade
            }

        results = {
            'Prompt-level-strict-accuracy':
            prompt_strict_correct / prompt_strict_total * 100,
            'Inst-level-strict-accuracy':
            inst_strict_correct / inst_strict_total * 100,
            'Prompt-level-loose-accuracy':
            prompt_loose_correct / prompt_loose_total * 100,
            'Inst-level-loose-accuracy':
            inst_loose_correct / inst_loose_total * 100,
            'details':
            details
        }
        return results
