from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class QASPERDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        import json
        import os
        dataset_dict = DatasetDict()
        split = 'dev'
        dev_list = []

        dev = os.path.join(path, 'qasper-dev-v0.3.json')
        with open(dev, 'r') as f:
            dev_json = json.load(f)

        for article_id in dev_json.keys():
            full_article = '\n'.join([
                (x['section_name'] if x['section_name'] else '') + '\n' +
                '\n'.join(x['paragraphs']) + '\n'
                for x in dev_json[article_id]['full_text']
            ])
            for qa in dev_json[article_id]['qas']:
                question = qa['question']
                answers = []
                for x in qa['answers']:
                    answers.extend(x['answer']['extractive_spans'])
                if answers:
                    dev_list.append({
                        'answer': answers,
                        'question': question,
                        'evidence': full_article,
                    })
                else:
                    continue
        dataset_dict[split] = Dataset.from_list(dev_list)
        return dataset_dict
