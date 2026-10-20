import json
import os
import re
from typing import Tuple

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

langs_dict = {
    'FR-FR': ['La réponse est', 'la réponse est'],
    'EN-US': ['the answer is', 'The answer is'],
    'VI-VT': ['Câu trả lời là', 'câu trả lời là'],
    'AR-XY': ['الجواب هو'],
    'TH-TL': ['คำตอบคือ'],
    'ZH-CN': ['答案是'],
    'KO-KR': ['답변은'],
    'PT-BR': ['A resposta é'],
    'JA-JP': ['答えは'],
    'ES-LA': ['La respuesta es']
}


def extract_choice(gen, lang):
    r"""{ "answer": "A|B|C|D" }"""
    patterns = [
        r"\{\s*?\"answer\"\s*?\:\s*?\"?(A|B|C|D).*?\"?\s*?\}",
        r"\{\s*?[\'\"]answer[\'\"]\s*?\:\s*?[\'\"](A|B|C|D).*?[\'\"]\s*?\}",
        r"\"answer\"\s*:\s*\"?(A|B|C|D)\"?",
        r"[\'\"]answer[\'\"]\s*:\s*[\'\"](A|B|C|D)[\'\"]"
    ]
    for pattern in patterns:
        res = re.findall(pattern, gen, flags=re.DOTALL)
        if len(res) >= 1:
            return res[-1]

    else:
        res = None
        pattern = langs_dict[lang]
        for p in pattern:
            if p in gen and p != gen:
                res = gen.split(p)
                if len(res) > 1 and len(res[-1].strip()) > 0:
                    res = res[-1].strip()[0]
                else:
                    res = None
                break

        temp = ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd']
        if res in temp:
            return res
        else:
            return None


def extract_choice_fuzzy(gen):
    options = ['A', 'B', 'C', 'D']
    for option in options:
        if option in gen:
            return option
    return None


@TEXT_POSTPROCESSORS.register_module('pmmeval_mmmlu')
def pmmeval_mmmlu_postprocess(text: str, lang_code: str) -> Tuple[str]:
    return text, lang_code


@LOAD_DATASET.register_module()
class PMMEvalMMMLUDataset(BaseDataset):

    @staticmethod
    def load(path: str, lang: str, difficulty: str):
        assert difficulty in [
            'easy', 'hard', 'all'
        ], '`difficulty` should be one choice among "easy", "hard", and "all"!'
        data_path = get_data_path(path)

        if os.environ.get('DATASET_SOURCE') == 'ModelScope':
            dataset_list = list()
            from modelscope import MsDataset
            if difficulty == 'easy' or difficulty == 'all':
                dataset_list.append(
                    MsDataset.load(dataset_name=data_path,
                                   subset_name='mmmlu',
                                   split=f'easy/test/mmlu_{lang}'))
            if difficulty == 'hard' or difficulty == 'all':
                dataset_list.append(
                    MsDataset.load(dataset_name=data_path,
                                   subset_name='mmmlu',
                                   split=f'hard/test/mmlu_{lang}'))
            # TODO: conbine two datasets
            dataset = dataset_list[0] + dataset_list[1] if len(
                dataset_list) == 2 else dataset_list[0]
        else:
            dataset = list()
            if difficulty == 'easy' or difficulty == 'all':
                filename = os.path.join(data_path,
                                        f'mmmlu/easy/test/mmlu_{lang}.jsonl')
                with open(filename, mode='r', encoding='utf-8') as f:
                    for line in f:
                        line = json.loads(line.strip())
                        dataset.append(line)
            if difficulty == 'hard' or difficulty == 'all':
                filename = os.path.join(data_path,
                                        f'mmmlu/hard/test/mmlu_{lang}.jsonl')
                with open(filename, mode='r', encoding='utf-8') as f:
                    for line in f:
                        line = json.loads(line.strip())
                        dataset.append(line)

            dataset = Dataset.from_list(dataset)

        return dataset


class PMMEvalMMMLUEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        all_results = list()
        for (pred, lang), ref in zip(predictions, references):
            answer = extract_choice(pred, lang)
            if answer is None:
                answer = extract_choice_fuzzy(pred)
            if answer is None:
                acc = 0.0
                failed = 1.0
            else:
                acc = 1.0 if ref.lower() == answer.lower() else 0.0
                failed = 0.0

            all_results.append({
                'acc':
                acc,
                'failed':
                failed,
                'extracted_answer':
                pred if pred else 'no answer'
            })

        final_result = {
            'accuracy':
            round(
                sum(x['acc'] for x in all_results) / len(all_results) * 100,
                2),
            'details':
            all_results
        }

        return final_result
