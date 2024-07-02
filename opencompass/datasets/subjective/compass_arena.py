from typing import Optional

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from .subjective_cmp import SubjectiveCmpDataset


@LOAD_DATASET.register_module()
class CompassArenaDataset(SubjectiveCmpDataset):

    def load(self,
             path: str,
             name: str,
             mode: Optional[str] = 'm2n',
             infer_order: Optional[str] = 'double',
             base_models: Optional = None,
             summarizer: Optional = None):
        dataset = list(super().load(path, name))
        creation_dataset = []
        for data in dataset:
            if 'reference' in data['others']:
                if data['others']['reference'] is not None:
                    data['ref'] = data['others']['reference']
                else:
                    data['ref'] = '满足用户需求，言之有理即可'
            else:
                data['ref'] = '满足用户需求，言之有理即可'
            creation_dataset.append(data)
        dataset = Dataset.from_list(creation_dataset)
        return dataset
