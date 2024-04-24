import re

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MGSMSDataset(BaseDataset):

    @staticmethod
    def load(path: str):

        src_lines = open(path, 'r', encoding='utf-8').readlines()

        data = {'question': [], 'answer': []}

        for lines in src_lines:
            data['question'].append(lines.split('\t')[0])
            data['answer'].append(lines.split('\t')[1])

        dataset = Dataset.from_dict({
            'question': data['question'],
            'answer': data['answer']
        })
        return dataset


# LANG_TO_ANSWER_PREFIX = {
#     "en": "Answer",
#     "bn": "উত্তর",
#     "de": "Antwort",
#     "es": "Respuesta",
#     "fr": "Réponse",
#     "ja": "答え",
#     "ru": "Ответ",
#     "sw": "Jibu",
#     "te": "సమాధానం",
#     "th": "คำตอบ",
#     "zh": "答案",
# }


def mgsm_zh_postprocess(text: str) -> str:
    answer_text = text.split('答案')[-1].strip()
    numbers = re.findall(r'\d+\.?\d*', answer_text.replace(',', ''))
    return numbers[-1].rstrip('.') if numbers else ''


def mgsm_bn_postprocess(text: str) -> str:
    answer_text = text.split('উত্তর')[-1].strip()
    numbers = re.findall(r'\d+\.?\d*', answer_text.replace(',', ''))
    return numbers[-1].rstrip('.') if numbers else ''


def mgsm_de_postprocess(text: str) -> str:
    answer_text = text.split('Antwort')[-1].strip()
    numbers = re.findall(r'\d+\.?\d*', answer_text.replace(',', ''))
    return numbers[-1].rstrip('.') if numbers else ''


def mgsm_en_postprocess(text: str) -> str:
    answer_text = text.split('Answer')[-1].strip()
    numbers = re.findall(r'\d+\.?\d*', answer_text.replace(',', ''))
    return numbers[-1].rstrip('.') if numbers else ''


def mgsm_es_postprocess(text: str) -> str:
    answer_text = text.split('Respuesta')[-1].strip()
    numbers = re.findall(r'\d+\.?\d*', answer_text.replace(',', ''))
    return numbers[-1].rstrip('.') if numbers else ''


def mgsm_fr_postprocess(text: str) -> str:
    answer_text = text.split('Réponse')[-1].strip()
    numbers = re.findall(r'\d+\.?\d*', answer_text.replace(',', ''))
    return numbers[-1].rstrip('.') if numbers else ''


def mgsm_ja_postprocess(text: str) -> str:
    answer_text = text.split('答え')[-1].strip()
    numbers = re.findall(r'\d+\.?\d*', answer_text.replace(',', ''))
    return numbers[-1].rstrip('.') if numbers else ''


def mgsm_ru_postprocess(text: str) -> str:
    answer_text = text.split('Ответ')[-1].strip()
    numbers = re.findall(r'\d+\.?\d*', answer_text.replace(',', ''))
    return numbers[-1].rstrip('.') if numbers else ''


def mgsm_sw_postprocess(text: str) -> str:
    answer_text = text.split('Jibu')[-1].strip()
    numbers = re.findall(r'\d+\.?\d*', answer_text.replace(',', ''))
    return numbers[-1].rstrip('.') if numbers else ''


def mgsm_te_postprocess(text: str) -> str:
    answer_text = text.split('సమాధానం')[-1].strip()
    numbers = re.findall(r'\d+\.?\d*', answer_text.replace(',', ''))
    return numbers[-1].rstrip('.') if numbers else ''


def mgsm_th_postprocess(text: str) -> str:
    answer_text = text.split('คำตอบ')[-1].strip()
    numbers = re.findall(r'\d+\.?\d*', answer_text.replace(',', ''))
    return numbers[-1].rstrip('.') if numbers else ''


class MGSM_Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        result = {'pass': 0, 'fail': 0}
        for index, (references_answer, predictions_answer) in enumerate(
                zip(references, predictions)):
            if references_answer == predictions_answer:
                result['pass'] += 1
            else:
                result['fail'] += 1

        result['score'] = float(result['pass'] /
                                (result['pass'] + result['fail'])) * 100
        final_result = {'Acc': result['score']}
        return final_result
