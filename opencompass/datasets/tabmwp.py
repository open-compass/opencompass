import json
import os.path as osp
import random
import re
from typing import List

import numpy as np
from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator.icl_hf_evaluator import AccEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


def get_table_text(problem):
    table = problem['table']
    title = problem['table_title']
    if title and len(title) > 0:
        table = f'[TITLE]: {title}\n{table}'
    return table


def get_question_text(problem, option_inds='ABCDEFGH'):
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


def get_choices(problem):
    return problem['choices']


def get_unit(problem):
    return problem['unit']


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace('\n', '\\n')
    return solution


def normalize_answer(text, unit):
    # ["1,000", "123", "3/4", "56.456", "$56.4", "-3", "-10.02", "-3/2"]

    text = re.sub(r'^[\$]', '', text)
    text = re.sub(r'[\,\.\,\/]$', '', text)

    result = re.match(r'^[-+]?[\d,./]+$', text)

    if result is not None:
        # is number?
        text = text.replace(',', '')
        result = re.match(r'[-+]?\d+$', text)

        if result is not None:
            number = int(text)
        elif '/' in text:
            nums = text.split('/')
            number = round(float(nums[0]) / float(nums[1]), 3)
        else:
            number = round(float(text), 3)
        number = str(number)
        number = re.sub(r'\.[0]+$', '', number)
        return number
    else:
        # is text
        if unit:
            text = text.replace(unit, '').strip()
        return text


def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    if ' ' in str1 or ' ' in str2:
        str1_split = str1.split(' ')
        str2_split = str2.split(' ')
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def extract_prediction(output, options=None, option_inds='ABCDEFGH'):

    # $\\frac{16}{95}$ -> 16/95
    output = re.sub(r'\$?\\frac\{([\d\.\,\-]+)\}\{([\d\.\,]+)\}\$?', r'\1/\2',
                    output)

    output = re.sub(r'(?<![AP]\.M)\.$', '', output)
    output = re.sub(r'(?<=\d)[\=](?=[\-\$\d])', ' = ', output)
    output = re.sub(r'\u2212', '-', output)

    # Multi-choice questions
    if options:
        patterns = [
            r'^\(([A-Za-z])\)$',  # "(b)", "(B)"
            r'^([A-Za-z])$',  # "b", "B"
            r'^([A-Za-z]). ',  # "b", "B"
            r'[Th]he answer is ([A-Z])',  # "The answer is B"
            r'^\(([A-Za-z])\) [\s\S]+$',  # "(A) XXXXX"
            r'[Th]he answer is \(([A-Za-z])\) [\s\S]+$'
        ]

        # have "X" in the output
        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                pred = res[0].upper()  # e.g., "B"
                if pred in option_inds:
                    ind = option_inds.index(pred)  # 1
                    if ind >= len(options):
                        random.seed(123)
                        ind = random.choice(range(len(options)))
                    prediction = options[ind]
                    return prediction

        # find the most similar options
        scores = [score_string_similarity(x, output) for x in options]
        max_idx = int(
            np.argmax(scores))  # json does not recognize NumPy data types
        prediction = options[max_idx]
        return prediction

    else:
        # free_text QA problems, numeric answer
        patterns = [
            r'[Th]he answer is ([\s\S]+)$',  # "The answer is XXXXX.",
            r'[Th]he table shows that ([\d\$\.\,\/\:]+) ',
            r' = ([\d\$\.\,\/\:]+)',  # "= $1.40"
            r'(?<= be| is) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "will be $1.40"
            r'(?<= are| was) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "are $1.40"
            r'(?<= were) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "are $1.40"
            r' ([\d\$\.\,\/\:]+ [AP]\.M\.)',  # 7:25 P.M.
            r'([\-\d\$\.\,\/\:]{0,}[\d]+)',  # 14.5
        ]

        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                prediction = res[-1].strip()
                if prediction.endswith('.') and '.M.' not in prediction:
                    prediction = prediction[:-1]
                return prediction

    return output


@ICL_EVALUATORS.register_module()
class TabMWPEvaluator(AccEvaluator):
    """Accuracy evaluator for TabMWP Dataset."""

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
        preds, golds = [], []
        for idx in range(len(references)):
            pred = predictions[idx]
            unit = references[idx]['unit']
            answer = references[idx]['answer']
            choices = references[idx]['choices']
            preds.append(
                normalize_answer(extract_prediction(pred, choices),
                                 unit).lower())
            golds.append(normalize_answer(answer, unit).lower())
        return super()._preprocess(preds, golds)


@LOAD_DATASET.register_module()
class TabMWPDataset(BaseDataset):
    # The TabMWP dataset contains 38,431 tabular math word problems.
    # Each question in TabMWP is aligned with a tabular context,
    # which is presented as an image, semi-structured text, and a-
    # structured table. There are two types of questions: free-text-
    # and multi-choice, and each problem is annotated with gold-
    # solutions to reveal the multi-step reasoning process.
    # To learn more about it, please follow:
    # https://github.com/lupantech/PromptPG/tree/main
    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
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
                    unit = get_unit(problem)
                    answer = get_answer(problem)
                    choices = get_choices(problem)
                    solution = get_solution_text(problem)
                    raw_data.append({
                        'question':
                        question,
                        'table':
                        table,
                        'test_elements': {
                            'answer': answer,
                            'unit': unit,
                            'choices': choices
                        },
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
