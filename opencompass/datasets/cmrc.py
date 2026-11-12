import json

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CMRCDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 将原始数据转换为所需的格式
        rows = []
        for index, paragraphs in enumerate(data['data']):
            for paragraph in paragraphs['paragraphs']:

                context = paragraph['context']

                for question in paragraph['qas']:
                    answers = question['answers']
                    unique_answers = list(set([a['text'] for a in answers]))
                    rows.append({
                        'context': context,
                        'question': question['question'],
                        'answers': unique_answers
                    })

        # 创建 Dataset
        dataset = Dataset.from_dict({
            'context': [row['context'] for row in rows],
            'question': [row['question'] for row in rows],
            'answers': [row['answers'] for row in rows]
        })

        return dataset


@ICL_EVALUATORS.register_module()
class CMRCEvaluator(BaseEvaluator):
    """Exact match using the CMRC 2018 normalization rules.

    See ``ymcui/cmrc2018/baseline/cmrc2018_evaluate.py`` on GitHub.
    Preserve the complete answer and internal whitespace, including ASCII
    periods and commas, while lowercasing and removing selected punctuation.
    """

    _PUNCTUATION_TABLE = str.maketrans('', '',
                                       '-:_*^/\\~`+=，。：？！“”；’《》·、「」（）－～『』')

    @classmethod
    def _normalize(cls, text):
        return text.lower().strip().translate(cls._PUNCTUATION_TABLE)

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        if not predictions:
            return {'error': 'predictions and references must not be empty'}

        details = []
        for prediction, answers in zip(predictions, references):
            pred = self._normalize(prediction)
            normalized_answers = [
                self._normalize(answer) for answer in answers
            ]
            details.append({
                'pred':
                pred,
                'answer':
                list(dict.fromkeys(normalized_answers + answers)),
                'correct':
                pred in normalized_answers,
            })

        score = sum(detail['correct']
                    for detail in details) / len(details) * 100
        return {'score': score, 'details': details}


@TEXT_POSTPROCESSORS.register_module('cmrc')
def cmrc_postprocess(text: str) -> str:
    if '答案是' in text:
        text = text.split('答案是')[1]
    return text
