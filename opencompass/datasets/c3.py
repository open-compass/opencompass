import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class C3Dataset(BaseDataset):

    @staticmethod
    def load(path: str):

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rows = []
        for _, row in enumerate(data):
            content = row[0]
            content_str = ' '.join(
                [''.join(paragraph) for paragraph in content])

            for question in row[1]:
                label = question['choice'].index(question['answer'])
                length = len(question['choice'])
                if length < 4:
                    fill_value = question['choice'][0]  # 以第一个值为填充值
                    fill_count = 4 - length  # 需要填充的数量
                    question['choice'] += [fill_value] * fill_count  # 填充

                rows.append({
                    'content': content_str,
                    'question': question['question'],
                    'choices': question['choice'],
                    'choice0': question['choice'][0],
                    'choice1': question['choice'][1],
                    'choice2': question['choice'][2],
                    'choice3': question['choice'][3],
                    'label': label
                })

        dataset = Dataset.from_dict({
            'content': [row['content'] for row in rows],
            'question': [row['question'] for row in rows],
            'choice0': [row['choice0'] for row in rows],
            'choice1': [row['choice1'] for row in rows],
            'choice2': [row['choice2'] for row in rows],
            'choice3': [row['choice3'] for row in rows],
            'choices': [row['choices'] for row in rows],
            'label': [row['label'] for row in rows]
        })
        return dataset


@LOAD_DATASET.register_module()
class C3Dataset_V2(BaseDataset):

    @staticmethod
    def load(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        data = []
        for line in raw:
            content = ''.join([''.join(paragraph) for paragraph in line[0]])
            for question in line[1]:
                label = question['choice'].index(question['answer'])
                label = 'ABCD'[label]
                while len(question['choice']) < 4:
                    question['choice'].append('[NULL]')
                data.append({
                    'content': content,
                    'question': question['question'],
                    'choice0': question['choice'][0],
                    'choice1': question['choice'][1],
                    'choice2': question['choice'][2],
                    'choice3': question['choice'][3],
                    'label': label
                })
        return Dataset.from_list(data)
