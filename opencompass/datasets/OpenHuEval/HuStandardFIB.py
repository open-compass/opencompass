import json
import os
import re

from datasets import Dataset, DatasetDict
from fuzzywuzzy import fuzz

from opencompass.openicl.icl_evaluator import BaseEvaluator

from ..base import BaseDataset


class HuStandardFIBDataset(BaseDataset):

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
            instruction = obj['instruction']
            questions = obj['questions']
            hu_specific_dim = obj['hu_specific_dim']
            tmp = obj
            new_obj = dict(instruction=instruction,
                           questions=questions,
                           hu_specific_dim=hu_specific_dim,
                           reference=tmp)
            out_dict_list.append(new_obj)
        dataset = Dataset.from_list(out_dict_list)
        return dataset


class HuStandardFIBEvaluator(BaseEvaluator):
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
            std_ans = [
                re.sub(r'#\d+#', '', ans).split(';')
                for ans in refer['answers']
            ]  # Remove "#0#" and "#1#", then split refer['formatted_std_ans']

            blank_total += len(std_ans)
            question_total += 1
            model_ans = []
            pred = pred.strip()
            match = re.search(r'\{.*?\}', pred, re.DOTALL)
            if not match:
                details[idx] = {
                    'reference': refer,
                    'model_ans': model_ans,
                    'gt': std_ans,
                    'prompt': prompt,
                    'raw_pred': pred,
                    'blank_wise_correctness': [False] * len(std_ans),
                    'question_wise_correctness': False,
                }
                continue

            json_str = match.group(0)
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

            elif isinstance(formatted_json_str, dict):
                data = formatted_json_str
                to_end_flag = True

            blank_wise_correctness = []
            if to_end_flag:
                is_question_correct = True
                model_ans = [
                    re.sub(r'#\d+#', '', ans).split(';')
                    for ans in data.get('answers', [])
                ]  # Preprocess model_ans in the same way as std_ans
                for ans_idx, ans_list in enumerate(std_ans):
                    if ans_idx >= len(model_ans):
                        is_question_correct = False
                        blank_wise_correctness.append(False)
                        continue

                    model_list = model_ans[ans_idx]
                    is_blank_correct = True
                    for ans in ans_list:
                        best_match = max(
                            model_list,
                            key=lambda model: fuzz.ratio(ans, model))
                        if fuzz.ratio(ans, best_match) > 70:  # check threshold
                            blank_correct += 1
                        else:
                            is_blank_correct = False
                            is_question_correct = False
                    blank_wise_correctness.append(is_blank_correct)

                question_correct += 1 if is_question_correct else 0
            else:
                is_question_correct = False

            details[idx] = {
                'reference': refer,
                'std_ans': std_ans,
                'model_ans': model_ans,
                'prompt': prompt,
                'raw_pred': pred,
                'blank_wise_correctness': blank_wise_correctness,
                'question_wise_correctness': is_question_correct,
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
