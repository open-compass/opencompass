# flake8: noqa
# yapf: disable
import csv
import json
from typing import List
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

try:
    from dingo.model import Model
    from dingo.model.modelres import ModelRes
    from dingo.io import MetaData, SummaryModel, ErrorInfo
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
        num = len(predictions)
        summary = SummaryModel(
            score=0,
            num_good=0,
            num_bad=0,
            total=num
        )
        error_info_list = []
        if num == 0:
            return summary.to_dict()

        eval_model = 'pretrain'
        rule_map = Model.rule_groups[eval_model]

        for i in range(num):
            data = MetaData(
                data_id=i,
                prompt=origin_prompt[i],
                content=predictions[i]
            )
            error_info = ErrorInfo(data_id=data.data_id, prompt=data.prompt, content=data.content)

            if_good = True
            for rule in rule_map:
                tmp: ModelRes = rule.eval(data)
                # 结果判断
                if tmp.error_status is False:
                    continue
                if_good = False
                e_t = tmp.error_type
                e_n = tmp.error_type + '-' + tmp.error_name
                e_r = tmp.error_reason
                if e_t not in error_info.error_type:
                    error_info.error_type.append(e_t)
                error_info.error_name.append(e_n)
                error_info.error_reason.append(e_r)
            if if_good == False:
                summary.num_bad += 1
                error_info_list.append(error_info)
                for e_t in error_info.error_type:
                    if e_t not in summary.error_type_ratio:
                        summary.error_type_ratio[e_t] = 1
                    else:
                        summary.error_type_ratio[e_t] += 1
                for e_n in error_info.error_name:
                    if e_n not in summary.error_name_ratio:
                        summary.error_name_ratio[e_n] = 1
                    else:
                        summary.error_name_ratio[e_n] += 1


        summary.num_good = summary.total - summary.num_bad
        summary.score = round(summary.num_good / summary.total, 6)
        for e_t in summary.error_type_ratio:
            summary.error_type_ratio[e_t] = round(summary.error_type_ratio[e_t] / summary.total, 6)
        for e_n in summary.error_name_ratio:
            summary.error_name_ratio[e_n] = round(summary.error_name_ratio[e_n] / summary.total, 6)
        return summary.to_dict()
