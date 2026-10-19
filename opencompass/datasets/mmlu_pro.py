# flake8: noqa
# yapf: disable

from datasets import load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

CHOICES=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

def _parse(item):

    s = ''
    item['answer_string'] = ''
    for i, opt in enumerate(item['options']):
        if opt == 'N/A':
            continue
        option = '{}. {}\n'.format(CHOICES[i], opt)
        s += option
        if item['answer'] == CHOICES[i]:
            item['answer_string'] = option

    item['options_str'] = s.strip()
    item['cot_content'] = item['cot_content'].removeprefix("A: Let's think step by step.").strip()
    return item


@LOAD_DATASET.register_module()
class MMLUProDataset(BaseDataset):

    @staticmethod
    def load(path: str, category: str):
        path = get_data_path(path)
        mmlu_pro = load_dataset(path)
        mmlu_pro = mmlu_pro.filter(lambda x: x['category'] == category)
        mmlu_pro = mmlu_pro.map(_parse)
        return mmlu_pro

class MMLUProBaseEvaluator(BaseEvaluator):

    def is_equal(self, pred, refer):
        try:
            refer_option, refer_string = refer.split('. ')
            if pred in CHOICES and refer_option == pred:
                return True
            elif refer_string.strip() == pred:
                return True
            else :
                return False
        except Exception:
            pass
        return False

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            i = i.split('\n')[0].strip()
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if self.is_equal(i, j):
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result
