# flake8: noqa: E501
import re
from collections import defaultdict

from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


def match_general_answer(s):
    temp = s[0]
    if temp in ['A', 'B', 'C', 'D']:
        return temp
    else:
        return None


def match_GPT4_answer(s):
    if result := re.findall('(?:选择：|Choice: )([ABCD])', s):
        return result[0]
    else:
        return None


@ICL_EVALUATORS.register_module()
class Corev2Evaluator_(BaseEvaluator):

    def __init__(self,
                 base_model,
                 compare_model,
                 judge_method='gpt4',
                 metric='win_rate'):
        self.base_model = base_model
        self.compare_model = compare_model
        self.metric = metric
        self.judge_method = judge_method

    def score(self, predictions, references):
        if self.judge_method == 'gpt4':
            predictions = [match_GPT4_answer(s) for s in predictions]
        else:
            predictions = [match_general_answer(s) for s in predictions]
        print(
            f'Among {len(predictions)} judgements, successfully extracted {len(predictions)-predictions.count(None)} judgements.'
        )
        win_both, half_draw, categories = defaultdict(float), defaultdict(
            float), defaultdict(float)
        for prediction, reference in zip(predictions, references):
            if prediction is not None:
                categories[reference['capability'].split('-')[0]] += 1
                winner = ''
                if prediction == 'A':
                    winner = reference['model1']
                elif prediction == 'B':
                    winner = reference['model2']
                elif prediction == 'C':
                    win_both[reference['capability'].split('-')[0]] += 1
                if self.base_model == winner:
                    half_draw[reference['capability'].split('-')[0]] += 1
                    win_both[reference['capability'].split('-')[0]] += 1
        for capability in categories:
            if capability not in half_draw:
                win_both[capability] = 0.0
                half_draw[capability] = 0.0
            else:
                win_both[capability] = round(
                    (win_both[capability] / categories[capability]) * 100, 2)
                half_draw[capability] = round(
                    (half_draw[capability] / categories[capability]) * 100, 2)
        scores = {'win_both': win_both, 'half_draw': half_draw}
        return scores
