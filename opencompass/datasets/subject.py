# flake8: noqa: E501
import json
import random

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SubInfer_Dataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        raw_data = []
        with open(path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            for problem in json_data:
                question = problem['question']
                reference_answer = problem['reference_answer']
                evaluating_guidance = problem['evaluating_guidance']
                capability = problem['capability']
                raw_data.append({
                    'question': question,
                    'judge': {
                        'question': question,
                        'reference_answer': reference_answer,
                        'evaluating_guidance': evaluating_guidance,
                        'capability': capability
                    }
                })
        dataset = Dataset.from_list(raw_data)
        return dataset


@LOAD_DATASET.register_module()
class SubJudge_Dataset(BaseDataset):

    @staticmethod
    def load(
        path: str,
        model1: str,
        path2: str,
        model2: str,
        mode='compare',
        random_order=True,
        random_seed=0,
    ):
        dataset = DatasetDict()
        raw_data = []
        if mode == 'compare':
            with open(path, 'r', encoding='utf-8') as f:
                json_data1 = json.load(f)
            with open(path2, 'r', encoding='utf-8') as f:
                json_data2 = json.load(f)
            random_generate = random.Random(random_seed)
            same_flag = 0
            for idx in json_data1:
                problem = json_data1[idx]
                answer1 = json_data1[idx]['prediction']
                answer2 = json_data2[idx]['prediction']
                if answer1 == answer2:
                    same_flag += 1
                    continue
                item = {}
                item['question'] = problem['gold']['question']
                item['reference_answer'] = problem['gold']['reference_answer']
                item['evaluating_guidance'] = problem['gold'][
                    'evaluating_guidance']
                item['capability'] = problem['gold']['capability']
                if random_order:
                    if random_generate.randint(0, 1) == 0:
                        item['answer1'] = answer1
                        item['model1'] = model1
                        item['answer2'] = answer2
                        item['model2'] = model2
                    else:
                        item['answer1'] = answer2
                        item['model1'] = model2
                        item['answer2'] = answer1
                        item['model2'] = model1
                else:
                    item['answer1'] = answer1
                    item['model1'] = model1
                    item['answer2'] = answer2
                    item['model2'] = model2
                raw_data.append({
                    'question':
                    item['question'],
                    'reference_answer':
                    item['reference_answer'],
                    'evaluating_guidance':
                    item['evaluating_guidance'],
                    'capability':
                    item['capability'],
                    'answer1':
                    item['answer1'],
                    'answer2':
                    item['answer2'],
                    'judge': {
                        'capability': item['capability'],
                        'model1': item['model1'],
                        'model2': item['model2']
                    }
                })
            if same_flag != 0:
                print(
                    f'Among {len(json_data1)} comparisons, {same_flag} cases are exact match, which will be skipped. '
                )
        elif mode == 'score':
            pass
        dataset = Dataset.from_list(raw_data)
        return dataset
