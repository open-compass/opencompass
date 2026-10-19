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
    'fr': ['La réponse est', 'la réponse est'],
    'en': ['the answer is', 'The answer is'],
    'vi': ['Câu trả lời là', 'câu trả lời là'],
    'ar': ['الجواب هو'],
    'th': ['คำตอบคือ'],
    'zh': ['答案是'],
    'ko': ['답변은'],
    'pt': ['A resposta é'],
    'ja': ['答えは'],
    'id': ['Jawaban adalah', 'jawaban adalah'],
    'es': ['La respuesta es']
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


def extract_choice_fuzzy(gen, lang):
    options = ['A', 'B', 'C', 'D']  # 定义选项
    for option in options:
        if option in gen:  # 检查选项是否在文本中
            return option  # 返回第一个出现的选项
    return None


@TEXT_POSTPROCESSORS.register_module('pmmeval_xnli')
def pmmeval_xnli_postprocess(text: str, lang_code: str) -> Tuple[str]:
    return text, lang_code


@LOAD_DATASET.register_module()
class PMMEvalXNLIDataset(BaseDataset):

    @staticmethod
    def load(path: str, lang: str):
        data_path = get_data_path(path)
        if os.environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(dataset_name=data_path,
                                     subset_name='xnli',
                                     split=f'test/{lang}')
        else:
            dataset = list()
            filename = os.path.join(data_path, f'xnli/test/{lang}.jsonl')
            with open(filename, mode='r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line.strip())
                    dataset.append(line)
            dataset = Dataset.from_list(dataset)

        return dataset


class PMMEvalXNLIEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        all_results = list()

        for (pred, lang), ref in zip(predictions, references):
            choice = extract_choice(pred, lang)
            acc = 0
            failed_strict = 0
            failed = 1
            if choice is not None:
                failed = 0
                if ref.lower() == choice.lower():
                    acc = 1
                else:
                    acc = 0
            else:
                choice = extract_choice_fuzzy(pred, lang)
                if choice is None:
                    acc = 0
                    failed_strict = 1
                else:
                    failed_strict = 0
                    if ref.lower() == choice.lower():
                        acc = 1
                    else:
                        acc = 0

            all_results.append({
                'acc':
                float(acc),
                'failed':
                float(failed),
                'failed_strict':
                float(failed_strict),
                'extracted_answer':
                choice if choice else 'no answer',
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
