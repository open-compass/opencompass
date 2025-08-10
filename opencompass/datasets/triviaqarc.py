from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class TriviaQArcDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        import json
        import os
        dataset_dict = DatasetDict()
        split = 'dev'
        dev_list = []

        web_dev = os.path.join(path, 'qa', 'verified-web-dev.json')
        with open(web_dev, 'r') as f:
            web_dev_json = json.load(f)

        for x in web_dev_json['Data']:
            cand_answers = x['Answer']['Aliases'] + x['Answer']['HumanAnswers']
            question = x['Question']
            evidence = ''
            if x['SearchResults']:
                x_path = os.path.join(path, 'evidence', 'web',
                                      x['SearchResults'][0]['Filename'])
                with open(x_path, 'r') as f:
                    evidence = f.read(100000)
            dev_list.append({
                'answer': cand_answers,
                'question': question,
                'evidence': evidence,
            })

        wiki_dev = os.path.join(path, 'qa', 'verified-wikipedia-dev.json')
        with open(wiki_dev, 'r') as f:
            wiki_dev_json = json.load(f)

        for x in wiki_dev_json['Data']:
            cand_answers = x['Answer']['Aliases']
            question = x['Question']
            evidence = ''
            if x['EntityPages']:
                x_path = os.path.join(path, 'evidence', 'wikipedia',
                                      x['EntityPages'][0]['Filename'])
                with open(x_path, 'r') as f:
                    evidence = f.read(100000)
            dev_list.append({
                'answer': cand_answers,
                'question': question,
                'evidence': evidence,
            })

        dataset_dict[split] = Dataset.from_list(dev_list)
        return dataset_dict
