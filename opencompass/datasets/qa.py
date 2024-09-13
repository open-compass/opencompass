# flake8: noqa
# yapf: disable
import os
import csv
import json
from typing import List
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

try:
    from dingo.model.model import Model
    from dingo.io import InputArgs
    from dingo.exec import Executor
except Exception:
    raise ModuleNotFoundError('=========== dingo register fail. please try: pip install dingo-python. ===========')

@LOAD_DATASET.register_module()
class qaDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        raw_data = []
        with open(path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row) < 1:
                    row = ['']
                raw_data.append({'input': row[0]})
        return Dataset.from_list(raw_data)


@LOAD_DATASET.register_module()
class qaLongDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        raw_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                raw_data.append({'input': json.loads(line).get('input')})
        return Dataset.from_list(raw_data)


@ICL_EVALUATORS.register_module()
class qaEvaluator(BaseEvaluator):

    def score(self, origin_prompt: List, predictions: List) -> dict:
        file_data = [{'prompt':pmt, 'prediction':prd} for pmt, prd in zip(origin_prompt, predictions)]
        file_name = 'tmp_file.txt'
        with open(file_name, 'a', encoding='utf-8') as f:
            for d in file_data:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')

        input_data = {
            "eval_models": ["pretrain"],
            "input_path": file_name,
            "output_path": "./output/dingo/",
            "dataset": "local",
            "datasource": "local",  # If not fill in this item, it will be the same as "dataset"
            "data_format": "jsonl",
            "column_prompt": ["prompt"],
            "column_content": ["prediction"],
        }
        Model.apply_config(input_data['custom_config_path'])
        input_args = InputArgs(**input_data)
        executor = Executor.exec_map["local"](input_args)
        result = executor.execute()
        summary = result[0].to_dict()

        os.remove(file_name)
        return summary