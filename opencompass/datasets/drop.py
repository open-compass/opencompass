import json

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class dropDataset(BaseDataset):

    @staticmethod
    def get_answers(validated_answers):
        answers = []
        for answer_item in validated_answers:
            if answer_item['number']:
                answers.append(answer_item['number'])
            elif any(answer_item['date'][i] for i in ['day', 'month', 'year']):
                d = [answer_item['date'][i] for i in ['day', 'month', 'year']]
                answers.append(' '.join(d).strip())
            else:
                for span in answer_item['spans']:
                    answers.append(span)
        answers = list(set(answers))
        return answers

    @staticmethod
    def load(path, only_number=True):
        with open(path, 'r', encoding='utf-8') as f:
            lines = json.load(f)
        dataset_list = []
        for line in lines.values():
            for qa_pair in line['qa_pairs']:
                validated_answers = qa_pair['validated_answers']
                if only_number and not any(i['number']
                                           for i in validated_answers):
                    continue
                item = {
                    'prompt': line['passage'],
                    'question': qa_pair['question'],
                    'answers': dropDataset.get_answers(validated_answers),
                }
                dataset_list.append(item)

        dataset_list = Dataset.from_list(dataset_list)
        return DatasetDict({'validation': dataset_list})
