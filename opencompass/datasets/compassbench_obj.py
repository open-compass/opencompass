import copy
import json
import re

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset


def get_number(options):

    result_string = ''
    for i, option in enumerate(options, start=65):
        result_string += f'{chr(i)}. {option}\n'
    return result_string


@LOAD_DATASET.register_module()
class CompassBenchObjectiveV1_3(BaseDataset):

    @staticmethod
    def load(path: str, name: str):

        circular_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']

        data = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as infile:
            for id, line in enumerate(infile):
                entry = json.loads(line)
                if 'cloze' in name:
                    data.append({
                        'question': entry['question'].strip(),
                        'answer': entry['answer'].strip(),
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
                        line['question'] = (line['question'].strip() + '\n' +
                                            get_number(line['options']))
                        data.append(line)
                else:
                    # treat as normal single choice question
                    entry['question'] = (entry['question'].strip() + '\n' +
                                         get_number(entry['options']))
                    data.append(entry)

        dataset = Dataset.from_list(data)
        return dataset


@LOAD_DATASET.register_module()
class CompassBenchObjectiveMath(BaseDataset):

    @staticmethod
    def load(path: str):
        with open(path, 'r') as infile:
            data = [json.loads(line) for line in infile]
            for idx in range(len(data)):
                item = data[idx]
                prefix = ''
                if item.get('question_type',
                            None) and item['question_type'] in [
                                'multiple-answer', '多选题'
                            ]:
                    if '_en_' in path:
                        prefix = 'This question may has multiple answers, \
please select all correct answers. like this: A, B, C as your final answer\n'

                    else:
                        prefix = '这道题可能有多个正确答案，请选择所有正确的答案，\
例如：A, B, C 作为你的最终答案\n'

                if item.get('options', None) and len(item['options']) != 0:
                    item['question'] = prefix + item[
                        'question'] + '\n' + get_number(item['options'])
        dataset = Dataset.from_list(data)
        return dataset


@TEXT_POSTPROCESSORS.register_module()
def compassbench_objective_v1_3_postprocess(text: str, name) -> str:
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
