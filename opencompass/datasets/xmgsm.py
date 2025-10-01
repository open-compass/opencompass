import re

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path
import json
from .base import BaseDataset


@LOAD_DATASET.register_module()
class xMGSMSDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        def read_jsonl(path: str):
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return data

        path = get_data_path(path, local_mode=True)
        # src_lines = open(path, 'r', encoding='utf-8').readlines()
        data = {'question': [], 'answer': []}
        all_data = read_jsonl(path)
        for lines in all_data:
            #question, answer = lines.strip().split('\t')
            data['question'].append(lines['question'])
            data['answer'].append(lines['answer_number'])

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
    'sr': 'Одговор',
    'ar': 'الإجاب',# based on above instructions
    'hu': 'Válasz',
    'cs': 'Odpověď',
    'vi': 'Câu trả lời',
    'ko': '답',
}


def mgsm_postprocess(text: str, lang: str) -> str:
    answer_prefix = LANG_TO_ANSWER_PREFIX[lang]
    if answer_prefix not in text:
        return ''
    answer_text = text.split(answer_prefix)[-1].strip()
    numbers = re.findall(r'\d+\.?\d*', answer_text.replace(',', ''))
    return numbers[-1].rstrip('.') if numbers else ''


class xMGSM_Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)
        print('references', references)
        print('predictions', predictions)

        num_correct, total = 0, 0
        details = {}
        for index, (references_answer, predictions_answer) in enumerate(
                zip(references, predictions)):
            if str(references_answer) == str(predictions_answer):
                is_correct = True
            else:
                is_correct = False

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
