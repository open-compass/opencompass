# flake8: noqa
"""KOR-Bench Evaluator."""

import json
import os
import re

from .icl_base_evaluator import BaseEvaluator


def read_json_or_jsonl(data_path, split='', mapping_key=None):
    base_path = os.path.join(data_path, split)
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    elif base_path.endswith('.json') or base_path.endswith('.jsonl'):
        file_path = base_path
    else:
        raise FileNotFoundError('No JSON or JSONL file found.')

    with open(file_path, 'r') as file:
        if file_path.endswith('.json'):
            data = json.load(file)
        elif file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in file]

    if mapping_key:
        return {
            item[mapping_key]: item
            for item in data if mapping_key in item
        }
    else:
        return data


def read_json_or_jsonl_with_idx(data_path, split='', idx=None):
    base_path = os.path.join(data_path, split)
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    elif base_path.endswith('.json') or base_path.endswith('.jsonl'):
        file_path = base_path
    else:
        raise FileNotFoundError('No JSON or JSONL file found.')

    with open(file_path, 'r', encoding='utf-8') as file:
        if file_path.endswith('.json'):
            data = json.load(file)
        elif file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in file]

    if idx is not None:
        try:
            return next(item for item in data if item.get('idx') == idx)
        except StopIteration:
            raise ValueError(f'No entry found for idx {idx}')
    else:
        return data


class korbenchEvaluator(BaseEvaluator):
    """Evaluator class for KOR-Bench tasks, inheriting from BaseEvaluator.

    This class implements the `score` method to evaluate the model's
    predictions against the reference answers, using the evaluation logic
    specific to KOR-Bench.
    """

    def __init__(self, question_type, mode):
        """Initialize the evaluator with question type and mode.

        Args:
            question_type (str): Type of questions (e.g., 'logic', 'operation', 'puzzle').
            mode (str): Evaluation mode (e.g., 'zero-shot', 'self-correction').
        """
        super().__init__()
        self.question_type = question_type
        self.mode = mode

        # Predefined index ranges for special evaluation cases
        self.idx_ranges = [
            [18],
            [73, 74, 77],
            [94],
            [115, 116, 117],
            [121, 122, 123, 125],
            [131, 132, 134, 135, 136],
            [141, 143, 149],
            list(range(145, 148)),
            list(range(151, 157)),
            [160, 161, 162],
            [164, 165, 166],
            [170],
            [206, 209],
            list(range(211, 216)),
            [217, 218],
        ]

    def score(self, predictions, references):
        """Evaluates the model's predictions against the references.

        Args:
            predictions (list): List of model predictions.
            references (list): List of reference answers (each reference is a dict).

        Returns:
            list: Evaluation results for each prediction.
        """
        if len(predictions) != len(references):
            return {
                'error': 'Predictions and references have different lengths'
            }

        data = []
        for idx, (prediction,
                  reference) in enumerate(zip(predictions, references)):
            record = {
                'idx': str(idx),
                'response': prediction,
                'answer': reference.get('answer'),
                'rule_id': reference.get('rule_id'),
                'question_type': self.question_type,
                # Include other necessary fields from reference if needed
            }
            data.append(record)

        results = self.evaluate_responses(data, self.question_type, self.mode)
        return results

    def evaluate_responses(self, data, question_type, mode):
        """Evaluates a list of responses.

        Args:
            data (list): List of records containing responses and answers.
            question_type (str): Type of questions.
            mode (str): Evaluation mode.

        Returns:
            list: List of evaluation results.
        """
        results = []
        for record in data:
            idx = record.get('idx')
            response = record.get('response')
            answer = record.get('answer')
            rule_id = record.get('rule_id')

            response_text = self.extract_text_from_brackets(response)
            is_correct = self.evaluate_response_vs_answer(
                response, answer, question_type, rule_id, idx)

            result_dict = {
                'idx': idx,
                'response': response,
                'response_text': response_text,
                'answer': answer,
                'is_correct': is_correct
            }
            results.append(result_dict)
        return results

    # Helper methods

    def extract_text_from_brackets(self, text, clean_level='basic'):
        """Extracts text enclosed in double brackets [[ ]].

        Args:
            text (str): The text to extract from.
            clean_level (str): The level of cleaning to perform.

        Returns:
            str: The extracted text or "NULL" if not found.
        """
        matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
        if not matches:
            matches = re.findall(r'\$\\boxed\{(.*?)\}\$', text, re.DOTALL)
        if not matches:
            matches = re.findall(r'\[\s*(.*?)\s*\]', text, re.DOTALL)
        if matches:
            match_str = matches[0].strip()
            if clean_level == 'clean':
                match_str = match_str.replace('"', '').replace(
                    '\n', '').replace(' ', '').replace('[',
                                                       '').replace(']', '')
            elif clean_level == 'logic':
                match_str = match_str.replace('"',
                                              '').replace('\n', '').replace(
                                                  ' ', '').replace('.', '')
            elif clean_level == 'math':
                match_str = match_str.replace('"', '').replace(
                    '\n', '').replace('[', '').replace(']',
                                                       '').replace('$', '')
                return f'{self.clean_latex(match_str)}'
            return f'[[{match_str}]]'
        return 'NULL'

    def clean_latex(self, latex_expr):
        """Cleans LaTeX expressions for parsing.

        Args:
            latex_expr (str): The LaTeX expression to clean.

        Returns:
            str: The cleaned expression.
        """
        if '=' in latex_expr:
            latex_expr = latex_expr.rsplit('=', 1)[1]
        latex_expr = re.sub(r'\\[()\[\]]', '', latex_expr)
        latex_expr = re.sub(r'\\text\{.*?\}', '', latex_expr)
        latex_expr = re.sub(r'\\(left|right|displaystyle)', '', latex_expr)
        latex_expr = latex_expr.replace('\\\\', '\\')
        return latex_expr

    def evaluate_response_vs_answer(self, response, answer, question_type,
                                    rule_id, idx):
        """Evaluates a single response against the answer.

        Args:
            response (str): The model's response.
            answer (str): The reference answer.
            question_type (str): The question type.
            rule_id (str): The rule ID.
            idx (str): The index of the question.

        Returns:
            bool: True if the response is correct, False otherwise.
        """
        if question_type == 'logic' and rule_id == '5':
            response_text = self.extract_text_from_brackets(response, 'logic')
            answer_text = self.extract_text_from_brackets(answer, 'logic')
            if response_text is None:
                return False
            normalized_response = self.rule5_normalize_content(response_text)
            normalized_answer = self.rule5_normalize_content(answer)
            return normalized_response == normalized_answer
        elif question_type == 'logic':
            response_text = self.extract_text_from_brackets(response, 'logic')
            answer_text = self.extract_text_from_brackets(answer, 'logic')
            return response_text == answer_text
        else:
            response_text = self.extract_text_from_brackets(response, 'clean')
            return response_text == answer

    def rule5_normalize_content(self, content):
        """Normalizes content for rule 5.

        Args:
            content (str): The content to normalize.

        Returns:
            list: Sorted list of content parts.
        """
        parts = [part.strip() for part in content.split(';')]
        sorted_parts = sorted(parts)
        return sorted_parts

    # Additional helper methods can be defined here
    # For example: methods to handle mathematical expressions, logic comparisons, etc.

    # Implement other helper functions as per your evaluation logic


# Example usage:
# evaluator = korbenchEvaluator(question_type='logic', mode='zero-shot')
# results = evaluator.score(predictions, references)
