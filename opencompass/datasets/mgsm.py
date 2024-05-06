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
            question, answer = lines.split('\t')
            data['question'].append(question)
            data['answer'].append(answer)

        dataset = Dataset.from_dict({
            'question': data['question'],
            'answer': data['answer']
        })
        return dataset


LANG_TO_ANSWER_PREFIX = {
    'en': 'Answer',
    'bn': 'উত্তর',
    'de': 'Antwort',
    'es': 'Respuesta',
    'fr': 'Réponse',
    'ja': '答え',
    'ru': 'Ответ',
    'sw': 'Jibu',
    'te': 'సమాధానం',
    'th': 'คำตอบ',
    'zh': '答案',
}


def mgsm_postprocess(text: str, lang: str) -> str:
    answer_prefix = LANG_TO_ANSWER_PREFIX[lang]
    if answer_prefix not in text:
        return ''
    answer_text = text.split(answer_prefix)[-1].strip()
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
        final_result = {'accuracy': result['score']}
        return final_result
