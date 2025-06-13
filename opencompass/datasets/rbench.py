from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class RBenchDataset(BaseDataset):

    @staticmethod
    def load_single(path, subset='en'):
        raw_data = []
        ds = load_dataset(path, f'rbench-t_{subset}')

        for data in ds['test']:
            raw_data.append({
                'RBench_Question_Input': data['question'],
                'RBench_Option_A': data['A'],
                'RBench_Option_B': data['B'],
                'RBench_Option_C': data['C'],
                'RBench_Option_D': data['D'],
                'RBench_Option_E': data['E'],
                'RBench_Option_F': data['F'],
                'target': data['answer'],
            })
        return Dataset.from_list(raw_data)

    @staticmethod
    def load(path, subset='en', **kwargs):
        test_dataset = RBenchDataset.load_single(path=path, subset=subset)
        return test_dataset


if __name__ == '__main__':
    dataset = RBenchDataset.load()
    print(dataset)
