import json
import re
from os import environ

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class GaokaoBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        data = get_data_path(path, local_mode=True)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            return MsDataset.load(path, subset_name=name, split='test')
        else:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            return Dataset.from_list(data['example'])


valid_gaokao_bench_question_types = [
    'single_choice', 'multi_choice', 'multi_question_choice',
    'five_out_of_seven', 'cloze', 'subjective', 'correction'
]


class GaokaoBenchEvaluator(BaseEvaluator):

    def __init__(self, question_type) -> None:
        super().__init__()
        assert question_type in valid_gaokao_bench_question_types
        self.question_type = question_type

    def do_predictions_postprocess(self, model_output, answer_lenth=None):
        if self.question_type == 'single_choice':
            model_answer = []
            temp = re.findall(r'[A-D]', model_output[::-1])
            if len(temp) != 0:
                model_answer.append(temp[0])

        elif self.question_type == 'multi_question_choice':
            model_answer = []
            temp = re.findall(r'【答案】\s*[:：]*\s*[A-Z]', model_output)

            if len(temp) == answer_lenth:
                for t in temp:
                    model_answer.append(re.findall(r'[A-Z]', t)[0])
            else:
                temp = re.findall(r'[A-Z]', model_output)
                if len(temp) > 0:
                    for k in range(min(len(temp), answer_lenth)):
                        model_answer.append(temp[k])

        elif self.question_type == 'multi_choice':
            model_answer = []
            answer = ''
            content = re.sub(r'\s+', '', model_output)
            answer_index = content.find('【答案】')
            if answer_index > 0:
                temp = content[answer_index:]
                if len(re.findall(r'[A-D]', temp)) > 0:
                    for t in re.findall(r'[A-D]', temp):
                        answer += t
            else:
                temp = content[-10:]
                if len(re.findall(r'[A-D]', temp)) > 0:
                    for t in re.findall(r'[A-D]', temp):
                        answer += t
            if len(answer) != 0:
                model_answer.append(answer)

        elif self.question_type == 'five_out_of_seven':
            model_answer = []
            temp = re.findall(r'[A-G]', model_output)
            if len(temp) > 0:
                for k in range(min(5, len(temp))):
                    model_answer.append(temp[k])

        return model_answer

    def ensure_same_length(self, pred, refr):
        if len(pred) == len(refr):
            return pred
        return ['Z'] * len(refr)

    def score(self, predictions, references):
        if self.question_type not in [
                'single_choice', 'multi_choice', 'multi_question_choice',
                'five_out_of_seven'
        ]:
            return {'score': 0}
        elif self.question_type == 'multi_choice':
            details = {}
            correct_score, total_score = 0, 0
            for index, (pred, refr) in enumerate(zip(predictions, references)):
                pred = self.do_predictions_postprocess(pred)
                pred = self.ensure_same_length(pred, refr)
                is_corrects = []
                for p, r in zip(pred, refr):
                    if p == r:
                        correct_score += 2
                        is_corrects.append(True)
                    else:
                        for i in p:
                            if i not in r:
                                break
                        else:
                            correct_score += 1
                        is_corrects.append(False)
                    total_score += 2
                details[str(index)] = {
                    'pred': pred,
                    'refr': refr,
                    'is_correct': all(is_corrects),
                }

        else:
            details = {}
            correct_score, total_score = 0, 0
            for index, (pred, refr) in enumerate(zip(predictions, references)):
                if self.question_type == 'multi_question_choice':
                    pred = self.do_predictions_postprocess(pred, len(refr))
                else:
                    pred = self.do_predictions_postprocess(pred)
                pred = self.ensure_same_length(pred, refr)
                is_corrects = []
                for p, r in zip(pred, refr):
                    is_correct = p == r
                    correct_score += is_correct
                    total_score += 1
                    is_corrects.append(is_correct)
                details[str(index)] = {
                    'pred': pred,
                    'refr': refr,
                    'is_correct': all(is_corrects),
                }
        return {'score': correct_score / total_score * 100, 'details': details}


for question_type in valid_gaokao_bench_question_types:
    # fix classic closure problem
    def _gaokao_register(question_type):
        ICL_EVALUATORS.register_module(
            name='GaokaoBenchEvaluator' + '_' + question_type,
            module=lambda *args, **kwargs: GaokaoBenchEvaluator(
                question_type=question_type, *args, **kwargs))

    _gaokao_register(question_type)
