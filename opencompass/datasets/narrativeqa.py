from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class NarrativeQADataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        import csv
        import os
        dataset_dict = DatasetDict()
        splits = ['train', 'valid', 'test']
        dataset_lists = {x: [] for x in splits}
        with open(os.path.join(path, 'qaps.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[1] == 'set':
                    continue
                split = row[1]  # set
                answers = [row[3], row[4]]  # row['answer1'], row['answer2']
                question = row[2]  # question
                x_path = os.path.join(path, 'tmp',
                                      row[0] + '.content')  # document_id

                try:
                    with open(x_path, 'r', encoding='utf-8') as f:
                        evidence = f.read(100000)
                except:  # noqa: E722
                    continue
                dataset_lists[split].append({
                    'answer': answers,
                    'question': question,
                    'evidence': evidence,
                })

        for split in splits:
            dataset_dict[split] = Dataset.from_list(dataset_lists[split])

        return dataset_dict
