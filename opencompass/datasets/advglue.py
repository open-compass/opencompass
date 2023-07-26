from datasets import load_dataset

from .base import BaseDataset


class ADVTripleDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):

        dataset = load_dataset(**kwargs)

        def preprocess(example):
            mapping = {0: 'A', 1: 'B', 2: 'C'}
            example['label_option'] = mapping[example['label']]
            return example

        dataset = dataset.map(preprocess)
        return dataset


class ADVBinaryDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):

        dataset = load_dataset(**kwargs)

        def preprocess(example):
            mapping = {0: 'A', 1: 'B'}
            example['label_option'] = mapping[example['label']]
            return example

        dataset = dataset.map(preprocess)
        return dataset


# label 0 for entailment, A. yes
# label 1 for neutral, B. maybe
# label 2 for contradiction, C. no
adv_mnliDataset = ADVTripleDataset
adv_mnli_mmDataset = ADVTripleDataset

# label 0 for entailment, A. yes
# label 1 for not entailment, B. no
adv_qnliDataset = ADVBinaryDataset
adv_rteDataset = ADVBinaryDataset

# label 0 for not_duplicate, A. no
# label 1 for duplicate, B. yes
adv_qqpDataset = ADVBinaryDataset

# label 0 for A. negative
# label 1 for B. positive
adv_sst2Dataset = ADVBinaryDataset
