import re

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MGSMSDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        src_lines = open(path, 'r', encoding='utf-8').readlines()
        data = {'question': [], 'answer': []}
        for lines in src_lines:
            question, answer = lines.strip().split('\t')
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


def _normalize_answer(answer: str) -> str:
    """Normalize a number the way openai/simple-evals ``score_mgsm`` does:
    strip trailing decimal zeros (``18.00`` -> ``18``) and drop thousands
    separators (``2,125`` -> ``2125``)."""
    if '.' in answer:
        answer = answer.rstrip('0').rstrip('.')
    return answer.replace(',', '')


class MGSM_Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        num_correct, total = 0, 0
        details = {}
        for index, (references_answer, predictions_answer) in enumerate(
                zip(references, predictions)):
            is_correct = (_normalize_answer(references_answer) ==
                          _normalize_answer(predictions_answer))

            num_correct += is_correct
            total += 1
            details[str(index)] = {
                'references': references_answer,
                'predictions': predictions_answer,
                'correct': is_correct,
            }

        accuracy = num_correct / total * 100
        final_result = {'accuracy': accuracy, 'details': details}
        return final_result
