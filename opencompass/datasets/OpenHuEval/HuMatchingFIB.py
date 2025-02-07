import json
import os
import re

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator

from ..base import BaseDataset


class HuMatchingFIBDataset(BaseDataset):

    @staticmethod
    def load(filepath):
        assert os.path.isfile(filepath)
        assert filepath.endswith('.jsonl')
        dataset = DatasetDict()
        f = open(filepath, 'r', encoding='utf-8')
        lines = f.readlines()
        objs = []
        for line in lines:
            obj = json.loads(line)
            objs.append(obj)
        out_dict_list = []
        for obj in objs:
            question = obj['question']
            options = obj['options']
            hu_specific_dim = obj['hu_specific_dim']
            tmp = obj
            new_obj = dict(question=question,
                           options=options,
                           hu_specific_dim=hu_specific_dim,
                           reference=tmp)
            out_dict_list.append(new_obj)
        dataset = Dataset.from_list(out_dict_list)
        return dataset


class HuMatchingFIBEvaluator(BaseEvaluator):
    """
    ref: opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator
    """

    def score(self, predictions, references, origin_prompt) -> dict:
        if len(predictions) != len(references):
            return {'error': 'preds and refers have different length.'}
        details = {}
        blank_correct, blank_total = 0, 0
        question_correct, question_total = 0, 0

        for idx, (pred, refer, prompt) in enumerate(
                zip(predictions, references, origin_prompt)):
            std_ans = refer['answer']
            model_ans = []
            pred = pred.strip()
            match = re.search(r'\{.*?\}', pred, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                blank_total += len(std_ans)
                question_total += 1
                details[idx] = {
                    'reference': refer,
                    'std_ans': std_ans,
                    'model_ans': model_ans,
                    'prompt': prompt,
                    'raw_pred': pred,
                    'blank_wise_correct': [False] * len(std_ans),
                    'question_wise_correct': False,
                }
                continue
            json_str = json_str.strip()
            json_str = json_str.replace('\\xa0', '')
            formatted_json_str = json_str

            to_end_flag = False
            if isinstance(formatted_json_str, str):
                try:
                    data = json.loads(formatted_json_str)
                    to_end_flag = True
                except json.JSONDecodeError:
                    print(f'Invalid JSON format. {idx}')
                    blank_total += len(std_ans)
                    question_total += 1

            elif isinstance(formatted_json_str, dict):
                data = formatted_json_str
                to_end_flag = True
            else:
                blank_total += len(std_ans)
                question_total += 1

            model_ans = []
            blank_wise_correct = []
            is_question_correct = True
            if to_end_flag:
                model_ans = data.get('answer', [])
                for index, ans in enumerate(std_ans):
                    if index >= len(model_ans):
                        is_question_correct = False
                        blank_wise_correct.append(False)
                        continue
                    if ans == model_ans[index]:
                        blank_correct += 1
                        blank_wise_correct.append(True)
                    else:
                        is_question_correct = False
                        blank_wise_correct.append(False)

                blank_total += len(std_ans)
                question_total += 1
                question_correct += 1 if is_question_correct else 0

            details[idx] = {
                'reference': refer,
                'std_ans': std_ans,
                'model_ans': model_ans,
                'prompt': prompt,
                'raw_pred': pred,
                'blank_wise_correct': blank_wise_correct,
                'question_wise_correct': is_question_correct,
            }
        results = {
            'blank_level_correctness':
            round(blank_correct / blank_total * 100, 2),
            'question_level_correctness':
            round(question_correct / question_total * 100, 2),
            'details':
            details
        }

        return results
