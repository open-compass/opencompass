import os

from datasets import load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class ClimaQADataset(BaseDataset):

    @staticmethod
    def load(path: str, task: str, **kwargs):

        path = get_data_path(path)
        path = os.path.join(path, task)
        climateqa = load_dataset(path)['train']

        input_column = []
        for i in range(len(climateqa)):
            if 'Options' in climateqa[i].keys(
            ) and climateqa[i]['Options'] is not None:
                input_column.append(climateqa[i]['Question'] + '\n' +
                                    climateqa[i]['Options'])
            else:
                input_column.append(climateqa[i]['Question'])
        climateqa = climateqa.add_column(name='input', column=input_column)
        climateqa = climateqa.rename_column('Answer', 'target')
        return climateqa
