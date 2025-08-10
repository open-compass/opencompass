import os

from datasets import load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class PHYSICSDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path)
        path = os.path.join(path, name)
        physics = load_dataset(path)['train']
        physics = physics.rename_column('questions', 'input')

        target = []
        for i in physics:
            this_final_answer = ''
            for j in range(len(i['final_answers'])):
                this_final_answer += 'Answer ' + str(j + 1) + ': '
                this_final_answer += i['final_answers'][j]
                this_final_answer += '\n'
            target.append(this_final_answer)
        physics = physics.add_column(name='target', column=target)

        return physics
