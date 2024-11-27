import json
import os
import re

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


def _get_last_digit(s):
    _PAT_LAST_DIGIT = re.compile(
        r'([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)'  # noqa E501
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        last_digit = match[-1].group().replace(',', '').replace(
            '+', '').strip().strip('.')
        # print(f"The last digit in {s} is {last_digit}")
    else:
        last_digit = None
        # logger.warning(f"No digits found in {s!r}")
    return last_digit


@LOAD_DATASET.register_module()
class PMMEvalMGSMDataset(BaseDataset):

    @staticmethod
    def load(path: str, lang: str):
        data_path = get_data_path(path)

        if os.environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(dataset_name=data_path,
                                     subset_name='mgsm',
                                     split=f'test/{lang}')
        else:
            dataset = list()
            filename = os.path.join(data_path, f'mgsm/test/{lang}.jsonl')
            with open(filename, mode='r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line.strip())
                    dataset.append(line)
            dataset = Dataset.from_list(dataset)

        return dataset


class PMMEvalMGSMEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        num_correct, total = 0, 0
        details = {}
        for index, (references_answer, predictions_answer) in enumerate(
                zip(references, predictions)):
            extracted_answer = _get_last_digit(predictions_answer)
            references_answer = references_answer.replace(',', '')
            if references_answer == extracted_answer:
                is_correct = True
            else:
                is_correct = False

            num_correct += is_correct
            total += 1
            details[str(index)] = {
                'references': references_answer,
                'predictions': predictions_answer,
                'extracted': extracted_answer,
                'correct': is_correct,
            }

        accuracy = round(num_correct / total * 100, 2)
        final_result = {'accuracy': accuracy, 'details': details}
        return final_result
