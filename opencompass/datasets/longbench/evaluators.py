import difflib
import re
import string
from collections import Counter
from typing import List

import jieba
from fuzzywuzzy import fuzz
from rouge import Rouge

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return ''.join(text.split())

    def remove_punc(text):
        cn_punctuation = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀\
            ｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'

        all_punctuation = set(string.punctuation + cn_punctuation)
        return ''.join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


@ICL_EVALUATORS.register_module()
class LongBenchF1Evaluator(BaseEvaluator):

    def __init__(self, language: str = 'en') -> None:
        super().__init__()
        assert language in ['en', 'zh']
        self.language = language

    def score(self, predictions: List, references: List) -> dict:

        def f1_score(prediction, reference, **kwargs):
            common = Counter(prediction) & Counter(reference)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction)
            recall = 1.0 * num_same / len(reference)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        score = 0.
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference_list = references[i]
            task_score = 0.
            for reference in reference_list:
                if self.language == 'en':
                    normalized_prediction = normalize_answer(prediction)
                    normalized_reference = normalize_answer(reference)

                    prediction_tokens = normalized_prediction.split()
                    reference_tokens = normalized_reference.split()

                else:
                    prediction_tokens = list(
                        jieba.cut(prediction, cut_all=False))
                    reference_tokens = list(jieba.cut(reference,
                                                      cut_all=False))
                    prediction_tokens = [
                        normalize_zh_answer(token)
                        for token in prediction_tokens
                    ]
                    reference_tokens = [
                        normalize_zh_answer(token)
                        for token in reference_tokens
                    ]
                    prediction_tokens = [
                        token for token in prediction_tokens if len(token) > 0
                    ]
                    reference_tokens = [
                        token for token in reference_tokens if len(token) > 0
                    ]

                task_score = max(task_score,
                                 f1_score(prediction_tokens, reference_tokens))

            score += task_score

        score = score / len(predictions) * 100
        return {'score': score}


@ICL_EVALUATORS.register_module()
class LongBenchCountEvaluator(BaseEvaluator):

    def score(self, predictions: List, references: List) -> dict:
        score = 0.
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference_list = references[i]
            for reference in reference_list:
                numbers = re.findall(r'\d+', prediction)
                right_num = 0
                for number in numbers:
                    if str(number) == str(reference):
                        right_num += 1
                score += 0.0 if len(numbers) == 0 else float(right_num /
                                                             len(numbers))

        score = score / len(predictions) * 100
        return {'score': score}


@ICL_EVALUATORS.register_module()
class LongBenchRetrievalEvaluator(BaseEvaluator):

    def __init__(self, language: str = 'en') -> None:
        super().__init__()
        assert language in ['en', 'zh']
        self.language = language

    def score(self, predictions: List, references: List) -> dict:
        score = 0.
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference_list = references[i]
            for reference in reference_list:
                if self.language == 'en':
                    pattern = r'Paragraph (\d+)'
                else:
                    pattern = r'段落(\d+)'

                matches = re.findall(pattern, reference)
                reference_id = matches[0]
                numbers = re.findall(r'\d+', prediction)
                right_num = 0
                for number in numbers:
                    if str(number) == str(reference_id):
                        right_num += 1

                score += 0.0 if len(numbers) == 0 else float(right_num /
                                                             len(numbers))

        score = score / len(predictions) * 100
        return {'score': score}


@ICL_EVALUATORS.register_module()
class LongBenchRougeEvaluator(BaseEvaluator):

    def __init__(self, language: str = 'en') -> None:
        super().__init__()
        assert language in ['en', 'zh']
        self.language = language

    def score(self, predictions: List, references: List) -> dict:
        score = 0.
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference_list = references[i]
            task_score = 0.
            for reference in reference_list:
                if self.language == 'zh':
                    prediction = ' '.join(
                        list(jieba.cut(prediction, cut_all=False)))
                    reference = ' '.join(
                        list(jieba.cut(reference, cut_all=False)))

                rouge = Rouge()
                try:
                    cur_score = rouge.get_scores([prediction], [reference],
                                                 avg=True)['rouge-l']['f']
                except Exception:
                    cur_score = 0.
                task_score = max(task_score, cur_score)

            score += task_score

        score = score / len(predictions) * 100
        return {'score': score}


@ICL_EVALUATORS.register_module()
class LongBenchCodeSimEvaluator(BaseEvaluator):

    def score(self, predictions: List, references: List) -> dict:
        score = 0.
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference_list = references[i]
            task_score = 0.
            for reference in reference_list:
                all_lines = prediction.lstrip('\n').split('\n')
                prediction = ''
                for line in all_lines:
                    if ('`' not in line) and ('#'
                                              not in line) and ('//'
                                                                not in line):
                        prediction = line
                        break
                task_score = max(task_score,
                                 (fuzz.ratio(prediction, reference) / 100))

            score += task_score

        score = score / len(predictions) * 100
        return {'score': score}


@ICL_EVALUATORS.register_module()
class LongBenchClassificationEvaluator(BaseEvaluator):

    def score(self, predictions: List, references: List) -> dict:
        score = 0.
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference_list = references[i]['answers']
            for reference in reference_list:
                em_match_list = []
                all_classes = references[i]['all_classes']
                for class_name in all_classes:
                    if class_name in prediction:
                        em_match_list.append(class_name)
                for match_term in em_match_list:
                    if match_term in reference and match_term != reference:
                        em_match_list.remove(match_term)
                if em_match_list != 0:
                    if reference in em_match_list:
                        score += (1.0 / len(em_match_list))
                else:
                    best_match = None
                    highest_similarity = 0
                    for names in all_classes:
                        similarity = difflib.SequenceMatcher(
                            None, names, prediction).ratio()
                        if similarity > highest_similarity:
                            highest_similarity = similarity
                            best_match = names
                    score += float(best_match == reference)

        score = score / len(predictions) * 100
        return {'score': score}
