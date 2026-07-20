# flake8: nodingo
# yapf: disable
import csv
import json
import os
import time
from typing import List

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class DingoDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        raw_data = []
        with open(path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row) < 1:
                    row = ['']
                raw_data.append({'input': row[0]})
        return Dataset.from_list(raw_data)


@LOAD_DATASET.register_module()
class DingoLongDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        raw_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                raw_data.append({'input': json.loads(line).get('input')})
        return Dataset.from_list(raw_data)


@ICL_EVALUATORS.register_module()
class DingoEvaluator(BaseEvaluator):

    @staticmethod
    def _build_input_args(input_path: str):
        try:
            from dingo.io import InputArgs
        except ImportError:
            from dingo.config import InputArgs

            return InputArgs(
                input_path=input_path,
                output_path='./outputs/dingo/',
                dataset={
                    'source': 'local',
                    'format': 'jsonl',
                    'field': {
                        'prompt': 'prompt',
                        'content': 'prediction',
                    },
                },
                executor={
                    'eval_group': 'llm_base',
                    'result_save': {
                        'bad': True,
                    },
                },
            )

        return InputArgs(
            eval_group='llm_base',
            input_path=input_path,
            output_path='./outputs/dingo/',
            save_data=True,
            dataset='local',
            data_format='jsonl',
            column_prompt='prompt',
            column_content='prediction',
        )

    @staticmethod
    def _summary_to_dict(result):
        if isinstance(result, list):
            result = result[0]
        if hasattr(result, 'to_dict'):
            return result.to_dict()
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        return dict(result)

    def score(self, origin_prompt: List, predictions: List) -> dict:
        try:
            from dingo.exec import Executor
        except Exception:
            raise ModuleNotFoundError(
                '=========== '
                'dingo register fail. please try: pip install dingo-python.'
                ' ===========')

        current_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        file_data = [{'prompt': pmt, 'prediction': prd}
                     for pmt, prd in zip(origin_prompt, predictions)]
        os.makedirs('tmp', exist_ok=True)
        file_name = os.path.join('tmp', 'dingo_file_' + current_time + '.jsonl')  # noqa: E501

        with open(file_name, 'a', encoding='utf-8') as f:
            for d in file_data:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')
        try:
            input_args = self._build_input_args(file_name)
            executor = Executor.exec_map['local'](input_args)
            result = executor.execute()
            summary = self._summary_to_dict(result)
        except Exception:
            raise
        finally:
            os.remove(file_name)
        return summary
