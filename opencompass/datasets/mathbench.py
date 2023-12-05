import copy
import json
import os.path as osp
import re

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


def get_number(options):
    result_string = ''
    for i, option in enumerate(options, start=ord('A')):
        result_string += f'{chr(i)}. {option}\n'
    return result_string


def get_circular_example(entry, id):
    """For given example, generate four circular examples."""
    # Only 4 options is supported for current circular eval.
    circular_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']
    data = []
    for c in circular_patterns:
        line = copy.deepcopy(entry)
        options = []
        for i in range(4):
            options.append(line['options'][ord(c[i]) - ord('A')])
        line['options'] = options
        line['answer'] = {
            c[0]: 'A',
            c[1]: 'B',
            c[2]: 'C',
            c[3]: 'D'
        }[line['answer']]
        line['answer'] = str(id) + '--' + line['answer'] + '--' + c
        line['question'] = line['question'].strip() + '\n' + get_number(
            line['options'])
        data.append(line)

    return data


@LOAD_DATASET.register_module()
class MathBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, with_circular: bool = True):
        """MathBenth Dataset.

        Args:
            path (str): Path of the mathbench dataset.
            name (str): Name of the target subset.
            with_circular (bool): Whether to create circular dataset for
                single choice question. Defaults to True.
        """
        data = []
        filename = osp.join(path, f'{name}.jsonl')
        with open(filename, 'r') as infile:
            for id, line in enumerate(infile):
                entry = json.loads(line)
                if 'cloze' in name:
                    data.append({
                        'question': entry['question'].strip(),
                        'answer': entry['answer'].strip()
                    })
                else:
                    if with_circular:
                        data.extend(get_circular_example(entry, id))
                    else:
                        question = entry['question'].strip(
                        ) + '\n' + get_number(entry['options'])
                        info = {
                            'question': question,
                            'answer': entry['answer'].strip()
                        }
                        # For PPL evaluation
                        for i in range(4):
                            info[chr(ord('A') +
                                     i)] = entry['options'][i].strip()
                        data.append(info)

        dataset = Dataset.from_list(data)
        return dataset


@TEXT_POSTPROCESSORS.register_module()
def mathbench_postprocess(text: str, name: str) -> str:
    split = False
    ans = text
    if '_cn' in name:
        ans_line = ans.split('答案是')
    else:
        ans_line = ans.split('The answer is')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
        split = True

    output = re.sub(r'(\d),(\d)', r'\1\2', ans)
    numbers = re.findall(r'-?\d*\.?/?\d+|\d+', output)

    if numbers:
        return numbers[0] if split else numbers[-1]

    return ans
