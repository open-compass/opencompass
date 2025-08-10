import math
import re
from datetime import datetime

import numpy as np
from datasets import load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


def check_correctness(answer: str, ground_truth, calid, upper_limit,
                      lower_limit):
    """"""
    calid = int(calid)

    if calid in [13, 68]:
        # Output Type: date

        if datetime.strptime(
                answer,
                '%m/%d/%Y').strftime('%-m/%-d/%Y') == datetime.strptime(
                    ground_truth, '%m/%d/%Y').strftime('%-m/%-d/%Y'):
            correctness = 1
        else:
            correctness = 0
    elif calid in [69]:
        # Output Type: integer (A, B)
        match = re.search(
            r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?"
            r"\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", ground_truth)
        ground_truth = f'({match.group(1)}, {match.group(3)})'
        match = re.search(
            r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?"
            r"\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", answer)
        if match:
            weeks = match.group(1)
            days = match.group(3)
            answer = f'({weeks}, {days})'
            if eval(answer) == eval(ground_truth):
                correctness = 1
            else:
                correctness = 0
        else:
            correctness = 0
    elif calid in [
            4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48,
            51, 69
    ]:
        # Output Type: integer A
        answer = round(eval(answer))
        if answer == eval(ground_truth):
            correctness = 1
        else:
            correctness = 0
    elif calid in [
            2, 3, 5, 6, 7, 8, 9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39,
            40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67
    ]:
        # Output Type: decimal
        answer = eval(answer)
        if answer >= eval(lower_limit) and answer <= eval(upper_limit):
            correctness = 1
        else:
            correctness = 0
    else:
        raise ValueError(f'Unknown calculator ID: {calid}')
    return correctness


def extract_answer(answer, calid):

    calid = int(calid)
    extracted_answer = re.findall(r'[Aa]nswer":\s*(.*?)\}', answer)
    matches = re.findall(
        r'"step_by_step_thinking":\s*"'
        r'([^"]+)"\s*,\s*"[Aa]nswer"', answer)

    if matches:
        # Select the last match
        last_match = matches[-1]
        explanation = last_match
    else:
        explanation = 'No Explanation'

    if len(extracted_answer) == 0:
        extracted_answer = 'Not Found'
    else:
        extracted_answer = extracted_answer[-1].strip().strip('"')
        if extracted_answer == 'str(short_and_direct\
                _answer_of_the_question)':
            extracted_answer = 'Not Found'
        if extracted_answer == 'str(value which is\
                the answer to the question)':
            extracted_answer = 'Not Found'
        if extracted_answer == 'X.XX':
            extracted_answer = 'Not Found'

    if calid in [13, 68]:
        # Output Type: date
        match = re.search(
            r'^(0?[1-9]|1[0-2])\/(0?[1-9]'
            r'|[12][0-9]|3[01])\/(\d{4})', extracted_answer)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            year = match.group(3)
            answer = f'{month:02}/{day:02}/{year}'
        else:
            answer = 'N/A'

    elif calid in [69]:
        # Output Type: integer (A, B)
        match = re.search(
            r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,"
            r"\?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer)
        extracted_answer = extracted_answer.replace('[', '(').replace(
            ']', ')').replace("'", '').replace('"', '')
        match = re.search(
            r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,"
            r"?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer)
        if match:
            weeks = match.group(1)
            days = match.group(3)
            answer = f'({weeks}, {days})'
        else:
            answer = 'N/A'
    elif calid in [
            4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48,
            51, 69
    ]:
        # Output Type: integer A
        match = re.search(r'(\d+) out of', extracted_answer)
        if match:  # cases like "3 out of 5"
            answer = match.group(1)
        else:
            match = re.search(r'-?\d+(, ?-?\d+)+', extracted_answer)
            if match:  # cases like "3, 4, 5"
                answer = str(len(match.group(0).split(',')))
            else:
                # match = re.findall(r"(?<!-)\d+", extracted_answer)
                match = re.findall(r'(-?\d+(\.\d+)?)', extracted_answer)
                # match = re.findall(r"-?\d+", extracted_answer)
                if len(match) > 0:  # find the last integer
                    answer = match[-1][0]
                    # answer = match[-1].lstrip("0")
                else:
                    answer = 'N/A'
    elif calid in [
            2, 3, 5, 6, 7, 8, 9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39,
            40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67
    ]:
        # Output Type: decimal
        match = re.search(r'str\((.*)\)', extracted_answer)
        if match:
            expression = match.group(1).replace('^', '**').replace(
                'is odd', '% 2 == 1').replace('is even', '% 2 == 0').replace(
                    'sqrt', 'math.sqrt').replace('.math', '').replace(
                        'weight',
                        '').replace('height', '').replace('mg/dl', '').replace(
                            'g/dl', '').replace('mmol/L', '').replace(
                                'kg', '').replace('g',
                                                  '').replace('mEq/L', '')
            expression = expression.split('#')[0]
            if expression.count('(') > expression.count(')'):  # add missing ')
                expression += ')' * (expression.count('(') -
                                     expression.count(')'))
            elif expression.count(')') > expression.count(
                    '('):  # add missing (
                expression = '(' * (expression.count(')') -
                                    expression.count('(')) + expression
            try:
                answer = eval(expression, {'__builtins__': None}, {
                    'min': min,
                    'pow': pow,
                    'round': round,
                    'abs': abs,
                    'int': int,
                    'float': float,
                    'math': math,
                    'np': np,
                    'numpy': np
                })
            except Exception:
                print(f'Error in evaluating expression: {expression}')
                answer = 'N/A'
        else:
            match = re.search(r'(-?\d+(\.\d+)?)\s*mL/min/1.73',
                              extracted_answer)
            if match:  # cases like "8.1 mL/min/1.73 m\u00b2"
                answer = eval(match.group(1))
            else:
                match = re.findall(r'(-?\d+(\.\d+)?)\%', extracted_answer)
                if len(match) > 0:  # cases like "53.1%"
                    answer = eval(match[-1][0]) / 100
                else:
                    match = re.findall(r'(-?\d+(\.\d+)?)', extracted_answer)
                    if len(
                            match
                    ) > 0:  # cases like "8.1 mL/min/1.73 m\u00b2" or "11.1"
                        answer = eval(match[-1][0])
                    else:
                        answer = 'N/A'
        if answer != 'N/A':
            answer = str(answer)

    return answer, explanation


def _parse(item, prompt_mode):
    item['row_number'] = item['Row Number']
    item['calculator_id'] = item['Calculator ID']
    item['calculator_name'] = item['Calculator Name']
    item['category'] = item['Category']
    item['output_type'] = item['Output Type']
    item['note_id'] = item['Note ID']
    item['note_type'] = item['Note Type']
    item['patient_note'] = item['Patient Note']
    item['question'] = item['Question']
    item['relevant_entities'] = item['Relevant Entities']
    item['ground_truth_answer'] = item['Ground Truth Answer']
    item['lower_limit'] = item['Lower Limit']
    item['upper_limit'] = item['Upper Limit']
    item['ground_truth_explanation'] = item['Ground Truth Explanation']
    return item


@LOAD_DATASET.register_module()
class MedCalc_BenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, prompt_mode: str, **kwargs):
        data_files = {
            'test': 'data/test-00000-of-00001.parquet',
            'train': 'data/train-00000-of-00001.parquet'
        }
        dataset = load_dataset(path, data_files=data_files, split='test')
        # dataset = dataset.select(range(2))
        if prompt_mode == 'zero-shot':
            dataset = dataset.map(lambda item: _parse(item, prompt_mode),
                                  load_from_cache_file=False)
        elif prompt_mode == 'few-shot':
            pass  # TODO: Implement few-shot prompt
        return dataset


class MedCalcOfficial_Evaluator(BaseEvaluator):

    def score(self, predictions, references, test_set):

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        correct = 0
        count = 0
        details = []
        for idx, (i, j) in enumerate(zip(predictions, references)):
            calculator_id = test_set['calculator_id'][idx]
            lower_limit = test_set['lower_limit'][idx]
            upper_limit = test_set['upper_limit'][idx]
            row_number = test_set['row_number'][idx]
            note_id = test_set['note_id'][idx]
            category = test_set['category'][idx]
            question = test_set['question'][idx]
            calculator_name = test_set['calculator_name'][idx]
            patient_note = test_set['patient_note'][idx]
            ground_truth_explanation = test_set['ground_truth_explanation'][
                idx]
            ground_truth_answer = test_set['ground_truth_answer'][idx]
            try:
                answer_value, explanation = extract_answer(
                    i, int(calculator_id))

                print(answer_value)
                print(explanation)

                correctness = check_correctness(answer_value,
                                                ground_truth_answer,
                                                calculator_id, upper_limit,
                                                lower_limit)

                status = 'Correct' if correctness else 'Incorrect'

                outputs = {
                    'Row Number': int(row_number),
                    'Calculator Name': calculator_name,
                    'Calculator ID': calculator_id,
                    'Category': category,
                    'Note ID': note_id,
                    'Patient Note': patient_note,
                    'Question': question,
                    'LLM Answer': answer_value,
                    'LLM Explanation': explanation,
                    'Ground Truth Answer': ground_truth_answer,
                    'Ground Truth Explanation': ground_truth_explanation,
                    'Result': status
                }

            except Exception as e:
                outputs = {
                    'Row Number': int(row_number),
                    'Calculator Name': calculator_name,
                    'Calculator ID': calculator_id,
                    'Category': category,
                    'Note ID': note_id,
                    'Patient Note': patient_note,
                    'Question': question,
                    'LLM Answer': str(e),
                    'LLM Explanation': str(e),
                    'Ground Truth Answer': ground_truth_answer,
                    'Ground Truth Explanation': ground_truth_explanation,
                    'Result': 'Incorrect'
                }
                status = 'Incorrect'
            count += 1
            if status == 'Correct':
                correct += 1
            details.append(outputs)

        result = {'accuracy': 100 * correct / count, 'details': details}
        return result
