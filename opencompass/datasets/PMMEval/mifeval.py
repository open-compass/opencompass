import json
import os
from typing import Tuple

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.datasets.PMMEval.mifeval_utils import mifeval_class_map
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path


def test_instruction_following_strict(inp, response, lang_code):
    """Tests response to see if instrutions are followed."""
    instruction_list = inp['instruction_id_list']
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_id_0, instruction_id_1 = tuple(instruction_id.split(':'))
        instruction_fuction_info = mifeval_class_map[instruction_id_0][
            instruction_id_1]

        instruction_function = instruction_fuction_info['function']
        instruction_function_args = dict()

        if instruction_fuction_info['required_lang_code']:
            instruction_function_args['lang_code'] = lang_code
        for kwarg_dict in inp['kwargs']:
            for k, v in kwarg_dict.items():
                if v is None:
                    continue
                instruction_function_args[k] = v
        instruction_function_args['input_string'] = response

        if response.strip() and instruction_function(
                **instruction_function_args):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return 1.0 if all(is_following_list) else 0.0


def test_instruction_following_loose(inp, response, lang_code):
    """Tests response for an upper bound for following instructions."""
    r = response.split('\n')
    response_remove_first = '\n'.join(r[1:]).strip()
    response_remove_last = '\n'.join(r[:-1]).strip()
    response_remove_both = '\n'.join(r[1:-1]).strip()
    revised_response = response.replace('*', '')
    revised_response_remove_first = response_remove_first.replace('*', '')
    revised_response_remove_last = response_remove_last.replace('*', '')
    revised_response_remove_both = response_remove_both.replace('*', '')
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp['instruction_id_list']
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_id_0, instruction_id_1 = tuple(instruction_id.split(':'))
        instruction_fuction_info = mifeval_class_map[instruction_id_0][
            instruction_id_1]

        instruction_function = instruction_fuction_info['function']
        instruction_function_args = dict()

        if instruction_fuction_info['required_lang_code']:
            instruction_function_args['lang_code'] = lang_code
        for kwarg_dict in inp['kwargs']:
            for k, v in kwarg_dict.items():
                instruction_function_args[k] = v
        instruction_function_args['input_string'] = response

        is_following = False
        for r in all_responses:
            if r.strip() and instruction_function(**instruction_function_args):
                is_following = True
                break

        is_following_list.append(is_following)

    return 1.0 if all(is_following_list) else 0.0


@TEXT_POSTPROCESSORS.register_module('pmmeval_mifeval')
def pmmeval_mifeval_postprocess(text: str, lang_code: str) -> Tuple[str]:
    return text, lang_code


@LOAD_DATASET.register_module()
class PMMEvalMIFEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str, lang: str):
        data_path = get_data_path(path)

        if os.environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(dataset_name=data_path,
                                     subset_name='mifeval',
                                     split=f'test/{lang}')
        else:
            dataset = list()
            filename = os.path.join(data_path, f'mifeval/test/{lang}.jsonl')
            with open(filename, mode='r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line.strip())
                    dataset.append(line)
            dataset = Dataset.from_list(dataset)

        return dataset


class PMMEvalMIFEvalEvaluator(BaseEvaluator):

    def score(self, predictions, references, test_set):
        all_results = list()
        for (pred, lang), example in zip(predictions, test_set):
            temp_result = {
                'strict_acc':
                test_instruction_following_strict(example, pred, lang),
                'loose_acc':
                test_instruction_following_loose(example, pred, lang)
            }

            all_results.append(temp_result)

        result = {
            'strict_acc':
            round(
                sum(x['strict_acc']
                    for x in all_results) / len(all_results) * 100, 2),
            'loose_acc':
            round(
                sum(x['loose_acc']
                    for x in all_results) / len(all_results) * 100, 2)
        }
        return result
