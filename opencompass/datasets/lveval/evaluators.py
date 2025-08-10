"""Functions for computing metrics.

Part of following code are modified from ` https://github.com/THUDM/LongBench`
"""

import re
import string
from collections import Counter
from typing import List

import jieba
from rouge import Rouge

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS

ABANDON_WORDS_EN = [
    'and',
    'to',
    'of',
    'in',
    'her',
    'was',
    'with',
    'for',
    'it',
    'from',
    'is',
    'that',
    'his',
    'he',
    'by',
    'she',
    'they',
    'or',
    'at',
    'because',
    'be',
    'on',
    'are',
    'their',
    'what',
    'as',
    'had',
    'were',
    'about',
    'being',
    'this',
    'who',
    'but',
    'have',
    'has',
    'when',
    'which',
    'does',
]

ABANDON_WORDS_ZH = [
    '的',
    '和',
    '是',
    '等',
    '在',
    '年',
    '可以',
    '为',
    '与',
    '‰',
    '了',
    '或',
    '一种',
    '月',
    'c',
    '至',
    '日',
    '有',
    '进行',
    '于',
    '不',
    '中',
    '×',
    '根据',
    '小',
    '由',
    '亩',
    '也',
    '要',
    '指',
    '法',
    '会',
    '元',
    '主要',
    '以及',
    '通过',
    '首先',
    '对',
    '然后',
    '号',
    '以',
    '所',
    '后',
    '丁',
    '包括',
    '无',
    '将',
    '用',
    '能',
    '形',
    '方面',
    '因素',
    '位于',
    '而',
    '从',
    '到',
    '一定',
    '用于',
    '但',
    '使用',
    '让',
    '具有',
    '并',
    '亿元',
    '万元',
    '上',
    '类',
    '基于',
    '才',
    '来',
    '地',
    '片',
    '其他',
    '个',
    '或者',
    '变得',
    '时',
    '给',
    '你',
    '使',
    '条',
    '受',
    '已经',
    '带',
    '度',
]


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
class LVEvalF1Evaluator(BaseEvaluator):

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

        score = 0.0
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference_list = references[i]
            task_score = 0.0
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
                break

            score += task_score

        score = score / len(predictions) * 100
        return {'f1': score}


@ICL_EVALUATORS.register_module()
class LVEvalOPTF1Evaluator(BaseEvaluator):

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

        score = 0.0
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference_list = references[i]
            answer_keyword = reference_list[-1]
            task_score = 0.0
            for reference in reference_list:
                if self.language == 'en':
                    normalized_prediction = normalize_answer(prediction)
                    normalized_reference = normalize_answer(reference)

                    prediction_tokens = normalized_prediction.split()
                    reference_tokens = normalized_reference.split()
                    # answer keywords recall
                    if answer_keyword:
                        answer_keyword_tokens = normalize_answer(
                            answer_keyword)
                        answer_keyword_tokens = answer_keyword_tokens.split()
                        common = Counter(prediction_tokens) & Counter(
                            answer_keyword_tokens)
                        filtered_common = {
                            key: value
                            for key, value in common.items()
                            if key not in ABANDON_WORDS_EN
                        }
                        num_same = sum(filtered_common.values())
                        recall = 1.0 * num_same / len(answer_keyword_tokens)
                        if recall < 0.2:
                            break
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
                    if not answer_keyword:
                        answer_keyword = reference
                    if answer_keyword:
                        answer_keyword_tokens = list(
                            jieba.cut(answer_keyword, cut_all=False))
                        answer_keyword_tokens = [
                            normalize_zh_answer(token)
                            for token in answer_keyword_tokens
                        ]
                        answer_keyword_tokens = [
                            token for token in answer_keyword_tokens
                            if len(token) > 0
                        ]
                        common = Counter(prediction_tokens) & Counter(
                            answer_keyword_tokens)
                        filtered_common = {
                            key: value
                            for key, value in common.items()
                            if key not in ABANDON_WORDS_ZH
                        }
                        num_same = sum(filtered_common.values())
                        recall = 1.0 * num_same / len(answer_keyword_tokens)
                        if recall < 0.4:
                            break

                task_score = max(task_score,
                                 f1_score(prediction_tokens, reference_tokens))
                break

            score += task_score

        score = score / len(predictions) * 100
        return {'LVEval_f1': score}


@ICL_EVALUATORS.register_module()
class LVEvalOPTRougeEvaluator(BaseEvaluator):

    def __init__(self, language: str = 'en') -> None:
        super().__init__()
        assert language in ['en', 'zh']
        self.language = language

    def score(self, predictions: List, references: List) -> dict:
        score = 0.0
        for i in range(len(predictions)):
            prediction = predictions[i]
            reference_list = references[i]
            task_score = 0.0
            for reference in reference_list:

                if self.language == 'zh':
                    word_blacklist = ABANDON_WORDS_ZH
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
                else:
                    word_blacklist = ABANDON_WORDS_EN
                    prediction_tokens = normalize_answer(prediction)
                    reference_tokens = normalize_answer(reference)
                    prediction_tokens = prediction_tokens.split()
                    reference_tokens = reference_tokens.split()

                filtered_prediction_tokens = [
                    i for i in prediction_tokens if i not in word_blacklist
                ]
                filtered_reference_tokens = [
                    i for i in reference_tokens if i not in word_blacklist
                ]
                prediction = ' '.join(filtered_prediction_tokens)
                reference = ' '.join(filtered_reference_tokens)

                rouge = Rouge()
                try:
                    cur_score = rouge.get_scores([prediction], [reference],
                                                 avg=True)['rouge-l']['f']
                except Exception:
                    cur_score = 0.0
                task_score = max(task_score, cur_score)
                break

            score += task_score

        score = score / len(predictions) * 100
        return {'LVEval_rouge': score}
