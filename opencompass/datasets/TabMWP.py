import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


def get_table_text(problem):
    table = problem['table']
    title = problem['table_title']
    if title and len(title) > 0:
        table = f'[TITLE]: {title}\n{table}'
    return table


def get_question_text(problem,
                      option_inds=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):
    question = problem['question']

    unit = problem['unit']
    if unit and len(unit) > 0:
        question = f'{question} (Unit: {unit})'

    choices = problem['choices']
    if choices and len(choices) > 0:
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append('({}) {}'.format(option_inds[i], c))
        options = ' '.join(choice_list)
        question = f'{question}\nOptions: {options}'

    return question


def get_answer(problem):
    return problem['answer']


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace('\n', '\\n')
    return solution


@LOAD_DATASET.register_module()
class TabMWPDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        for split in ['dev', 'test', 'train']:
            raw_data = []
            filename = osp.join(path, f'problems_{split}.json')
            with open(filename, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for idx in json_data:
                    problem = json_data[idx]
                    question = get_question_text(problem)
                    table = get_table_text(problem)
                    answer = get_answer(problem)
                    solution = get_solution_text(problem)
                    raw_data.append({
                        'question':
                        question,
                        'table':
                        table,
                        'answer':
                        f'Answer: The answer is {answer}.',
                        'solution':
                        f'Solution: {solution}',
                        'answer_and_solution':
                        f'Answer: The answer is {answer}. BECAUSE: {solution}',
                        'solution_and_answer':
                        f'Answer: {solution} The answer is {answer}.'
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset
