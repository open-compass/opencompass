import copy
import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


def get_number(options):

    result_string = ''
    for i, option in enumerate(options, start=65):
        result_string += f'{chr(i)}. {option}\n'
    return result_string


@LOAD_DATASET.register_module()
class WikiBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path, local_mode=True)

        circular_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']

        data = []
        with open(path, 'r', encoding='utf-8') as infile:
            for id, line in enumerate(infile):
                entry = json.loads(line)
                if 'cloze' in name:
                    data.append({
                        'question': entry['question'].strip(),
                        'answer': entry['answer'].strip()
                    })
                elif 'circular' in name:
                    for c in circular_patterns:
                        line = copy.deepcopy(entry)
                        options = []
                        for i in range(4):
                            options.append(line['options'][ord(c[i]) -
                                                           ord('A')])
                        line['options'] = options
                        line['answer'] = {
                            c[0]: 'A',
                            c[1]: 'B',
                            c[2]: 'C',
                            c[3]: 'D'
                        }[line['answer']]
                        line['answer'] = str(
                            id) + '--' + line['answer'] + '--' + c
                        line['question'] = line['question'].strip(
                        ) + '\n' + get_number(line['options'])
                        data.append(line)
                else:
                    # treat as normal single choice question
                    entry['question'] = entry['question'].strip(
                    ) + '\n' + get_number(entry['options'])
                    data.append(entry)

        dataset = Dataset.from_list(data)
        return dataset
