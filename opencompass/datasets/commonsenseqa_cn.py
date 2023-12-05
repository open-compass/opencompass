import json

from datasets import Dataset, DatasetDict

from .base import BaseDataset


class CommonsenseQADataset_CN(BaseDataset):

    @staticmethod
    def load(path):
        datasetdict = DatasetDict()
        for split in ['train', 'validation']:
            data = []
            with open(path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    data.append(item)

            def pre_process(example):
                for i in range(5):
                    example[chr(ord('A') + i)] = example['choices']['text'][i]
                return example

            dataset = Dataset.from_list(data)
            dataset = dataset.map(pre_process).remove_columns(
                ['question_concept', 'id', 'choices'])
            datasetdict[split] = dataset

        return datasetdict
