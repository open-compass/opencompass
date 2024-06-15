import datasets
from datasets import Dataset, DatasetDict
import json
import os
import re
from .base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


@LOAD_DATASET.register_module()
class CFinBench(BaseDataset):
    @staticmethod
    def load(path: str, name: str, data_type: str) -> Dataset:
        data_dict = {}
        for _set in ["dev", "test", "val"]:
            filepath = os.path.join(path, _set, data_type, name)
            with open(filepath, encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if data_type in ['single_choice', 'multi_choice']:
                        for option in data["OptionList"]:
                            data["text"] = data["text"] + "\n" + option
                    data.setdefault("explanation", '')
                    data_dict.setdefault(_set, []).append(data)
        dataset = {i: Dataset.from_list(data_dict[i]) for i in data_dict}
        return DatasetDict(dataset)


valid_data_types = ["single_choice", "multi_choice", "judgment", 'multi_choice-cot']


class CFinBenchEvaluator(BaseEvaluator):

    def __init__(self, data_type) -> None:
        super().__init__()
        self.data_type = data_type

    def postprocess(self, string: str):
        result = ""
        string = re.sub(r'[^\w\s]', '', string)
        if self.data_type == "multi_choice":
            content = re.sub(r'\s+', '', string)
            match = re.search(r'([A-E]+)', content)
            if match:
                result = match.group(1)
        elif self.data_type == "single_choice":
            content = re.sub(r'\s+', '', string)
            for t in content:
                if t.isupper():
                    result = t
                    break
        elif self.data_type == "judgment":
            content = re.sub(r'\s+', '', string)
            match = re.search(r'(正确|错误)', content)
            if match:
                result = match.group(1)
        return result

    def cot_postprocess(self, string: str):
        result = ""
        string = re.sub(r'[^\w\s]', '', string)
        if self.data_type == "multi_choice-cot":
            content = re.sub(r'\s+', '', string)
            pattern1 = r'答案.*?([A-E]+)'
            pattern2 = r'([A-E]{2,})'
            match = re.findall(pattern1, content)
            if match:
                result = match[0]
            else:
                match = re.findall(pattern2, content)
                if match:
                    result = match[0]
        return result

    def score(self, predictions: list, references: list) -> dict:
        if self.data_type not in valid_data_types:
            return {'score': 100}
        elif self.data_type == "multi_choice":
            correct_score, total_score = 0, 0
            for pred, ref in zip(predictions, references):
                pred = self.postprocess(pred)
                ref = self.postprocess(ref)
                if pred == ref:
                    correct_score += 2
                else:
                    for i in pred:
                        if i not in ref:
                            break
                    else:
                        correct_score += 1
                total_score += 2
            return {'score': correct_score / total_score * 100}
        elif self.data_type == 'multi_choice-cot':
            correct_score, total_score = 0, 0
            for pred, ref in zip(predictions, references):
                ref = self.cot_postprocess(ref)
                pred = self.cot_postprocess(pred)
                if pred == ref:
                    correct_score += 2
                else:
                    for i in pred:
                        if i not in ref:
                            break
                    else:
                        correct_score += 1
                total_score += 2
            return {'score': correct_score / total_score * 100}
        else:
            correct_score, total_score = 0, 0
            for pred, ref in zip(predictions, references):
                pred = self.postprocess(pred)
                ref = self.postprocess(ref)
                if pred == ref:
                    correct_score += 1
                total_score += 1
            return {'score': correct_score / total_score * 100}
