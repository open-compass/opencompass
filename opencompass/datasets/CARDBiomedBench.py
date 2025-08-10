from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


def _parse(item, prompt_mode):
    item['expert'] = item['Bio_Category']
    item['start'] = chr(65)
    item['end'] = chr(65 + len(item.get('choices', {'label': []})['label']) -
                      1)
    item['prompt_mode'] = prompt_mode
    return item


@LOAD_DATASET.register_module()
class CARDBiomedBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, prompt_mode: str, **kwargs):
        data_files = {'test': 'data/CARDBiomedBench.csv'}
        dataset = load_dataset(path, data_files=data_files, split='test')
        # dataset = dataset.select(range(200))
        if prompt_mode == 'zero-shot':
            dataset = dataset.map(lambda item: _parse(item, prompt_mode),
                                  load_from_cache_file=False)
        elif prompt_mode == 'few-shot':
            pass  # TODO: Implement few-shot prompt
        return dataset
