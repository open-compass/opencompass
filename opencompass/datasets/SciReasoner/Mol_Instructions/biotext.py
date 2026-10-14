# flake8: noqa
# molecule task
# https://github.com/zjunlp/Mol-Instructions/tree/main/evaluation/molecule

import json
import os
import re
from typing import List

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from sklearn.metrics import precision_recall_fscore_support

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path


def CER_calculate_f1_score(true_entities, predicted_entities):
    true_entities = set(true_entities.split(', '))
    predicted_entities = set(predicted_entities.split(', '))
    true_positive = len(true_entities & predicted_entities)
    precision = true_positive / len(predicted_entities) if len(
        predicted_entities) > 0 else 0
    recall = true_positive / len(true_entities) if len(
        true_entities) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (
        precision + recall) > 0 else 0
    # print(true_entities,predicted_entities,f1_score)
    return f1_score


def calculate_f1_score(true_entities, predicted_entities):
    # import pdb;pdb.set_trace()
    pattern = r'\(.*?\)'
    true_entities = re.findall(pattern, true_entities)
    predicted_entities_tmp = re.findall(pattern, predicted_entities)
    if not predicted_entities_tmp:
        # add () to predicted_entities if it is empty
        predicted_entities = f'({predicted_entities})'
        predicted_entities_tmp = re.findall(pattern, predicted_entities)

    predicted_entities = [entity.strip() for entity in predicted_entities_tmp]

    true_entities = set(true_entities)
    predicted_entities = set(predicted_entities)
    true_positive = len(true_entities & predicted_entities)
    precision = true_positive / len(predicted_entities) if len(
        predicted_entities) > 0 else 0
    recall = true_positive / len(true_entities) if len(
        true_entities) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (
        precision + recall) > 0 else 0
    return f1_score


def calculate_accuracy_(predictions, references):
    correct_count = 0
    total_count = len(references)
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred = pred[0].lower()
        ref = ref[0].lower()
        f1_score = calculate_f1_score(ref, pred)
        correct_count += f1_score

    return correct_count / total_count


def CER_calculate_accuracy_(predictions, references):
    correct_count = 0
    total_count = len(references)
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred = pred[0].lower()
        ref = ref[0].lower()
        f1_score = CER_calculate_f1_score(ref, pred)
        # print(f1_score)
        correct_count += f1_score

    return correct_count / total_count


def ture_or_false_calculate_accuracy_(predictions, references):
    x, y, z = 0, 0, 0
    correct_count = 0
    total_count = len(references)
    other_answers = 0
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred = pred[0].lower()
        ref = ref[0].lower()
        correct_first_word = ref.split(',')[0].strip().lower()
        # my_first_word = pred.split(',')[0].strip().lower()
        pred = pred.strip().lower()
        if 'yes' in pred:
            my_first_word = 'yes'
        elif 'no' in pred:
            my_first_word = 'no'
        elif 'maybe' in pred or 'may be' in pred or 'might' in pred:
            my_first_word = 'maybe'
        else:
            other_answers += 1
            my_first_word = 'other'
            print(f'Other answer: {pred}, reference: {ref}')

        if correct_first_word == 'no' and my_first_word == 'no':
            x += 1
        if correct_first_word == 'no':
            y += 1
        if my_first_word == 'no':
            z += 1
        if correct_first_word == my_first_word:
            correct_count += 1
    accuracy = (correct_count / total_count) * 100
    return accuracy, other_answers


def calculate_macro_f1_(predictions, references):
    correct_answers = [
        ref[0].split(',')[0].strip().lower() for ref in references
    ]
    my_answers = [
        pred[0].split(',')[0].strip().lower() for pred in predictions
    ]
    # Compute precision, recall, and F1-score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_answers,
        my_answers,
        labels=['yes', 'no', 'maybe'],
        average=None)
    # Calculate macro F1 by averaging F1-scores for all classes
    macro_f1 = sum(f1) / len(f1)

    return macro_f1


def multi_choice_question_calculate_accuracy(question_data):
    correct_count = 0
    total_count = len(question_data)
    for i, question in enumerate(question_data):
        correct_answer = question['output'].split('(')[1].split(')')[0]
        my_answer = question['my_output'][0]
        if '(A' in question['my_output'] or 'A)' in question[
                'my_output'] or ' A ' in question['my_output']:
            my_answer = 'A'
        elif '(B' in question['my_output'] or 'B)' in question[
                'my_output'] or ' B ' in question['my_output']:
            my_answer = 'B'
        elif '(C' in question['my_output'] or 'C)' in question[
                'my_output'] or ' C ' in question['my_output']:
            my_answer = 'C'
        elif '(D' in question['my_output'] or 'D)' in question[
                'my_output'] or ' D ' in question['my_output']:
            my_answer = 'D'
        if correct_answer == my_answer:
            correct_count += 1
    accuracy = (correct_count / total_count) * 100

    return accuracy


def multi_choice_question_calculate_accuracy_(predictions, references):
    correct_count = 0
    total_count = len(references)
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        correct_answer = ref[0].split('(')[1].split(')')[0]
        my_answer = pred[0]
        if '(A' in pred[0] or 'A)' in pred[0] or ' A ' in pred[0]:
            my_answer = 'A'
        elif '(B' in pred[0] or 'B)' in pred[0] or ' B ' in pred[0]:
            my_answer = 'B'
        elif '(C' in pred[0] or 'C)' in pred[0] or ' C ' in pred[0]:
            my_answer = 'C'
        elif '(D' in pred[0] or 'D)' in pred[0] or ' D ' in pred[0]:
            my_answer = 'D'
        if correct_answer == my_answer:
            correct_count += 1
    accuracy = (correct_count / total_count) * 100

    return accuracy


@LOAD_DATASET.register_module()
class Mol_Instructions_Dataset_BioText(BaseDataset):

    @staticmethod
    def load(path, task, max_cut=-1, mini_set=False, hf_hub=False):

        # if (hf_hub is True):
        #     # load from huggingface hub
        #     train_data = []
        #     repo_id = test_path.split('/')[0] + '/' + test_path.split('/')[1]
        #     train_path = train_path.split(repo_id + '/')[1]
        #     test_path = test_path.split(repo_id + '/')[1]
        #
        #     train_path = hf_hub_download(repo_id,
        #                                  train_path,
        #                                  repo_type='dataset')
        #     test_path = hf_hub_download(repo_id,
        #                                 test_path,
        #                                 repo_type='dataset')

        path = get_data_path(path)
        train_path = os.path.join(path, f'{task}/dev/data.json')
        test_path = os.path.join(path, f'{task}/test/data.json')

        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        train_data = train_data[:5]
        # Limit the dataset to 5 samples for testing purposes

        if (max_cut != -1):
            test_data = test_data[:max_cut]
        if mini_set:
            import random
            random.seed(1024)
            test_data = random.sample(test_data, min(len(test_data), 150))
            random.seed()

        dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'test': Dataset.from_list(test_data)
        })
        return dataset


@TEXT_POSTPROCESSORS.register_module('Mol_Instructions_postprocess_BioText')
def Mol_Instructions_postprocess_BioText(text, task, *args, **kwargs):
    """
    Extract the protein str between  <protein> and </protein> in the sentences
    """
    text = text.strip()
    if task in (
            'chemical_disease_interaction_extraction',
            'chemical_protein_interaction_extraction',
            'chemical_entity_recognition',
            'true_or_false_question',
            'multi_choice_question',
            'open_question',
    ):
        # For property prediction, we only need the first line of the text
        text = text.strip()
        text = re.sub(r'<\|endoftext\|>', '', text)
        text = re.sub(r'<\|im_end\|>', '', text)

        # remove "Response: " or "Answer: " at the beginning for qwen3
        text = re.sub(r'^(Response:|Answer:)\s*',
                      '',
                      text,
                      flags=re.IGNORECASE)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # remove the sentences before </think> for gpt-oss-120b
        text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)

        # remove "I would say that" or
        # "I would like to say that" at the beginning for qwen3
        text = re.sub(r'^(I would say that|I would like to say that)\s*',
                      '',
                      text,
                      flags=re.IGNORECASE)
        text = text.strip()
    else:
        pass
    return text


class Mol_Instructions_Evaluator_BioText(BaseEvaluator):

    def __init__(self, task='protein_design', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task

    def score(self, predictions: List[str], references: List[str]):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        if not isinstance(predictions[0], list):
            predictions = [[pred] for pred in predictions]
        if not isinstance(references[0], list):
            references = [[ref] for ref in references]

        if self.task in (
                'chemical_disease_interaction_extraction',
                'chemical_protein_interaction_extraction',
        ):
            results = {
                'f1': calculate_accuracy_(predictions, references),
            }
        elif self.task in ('chemical_entity_recognition', ):
            results = {
                'f1': CER_calculate_accuracy_(predictions, references),
            }
        elif self.task == 'true_or_false_question':
            acc, other_answers = ture_or_false_calculate_accuracy_(
                predictions, references)
            results = {
                'accuracy': acc,
                'other_answers': other_answers,
            }
        elif self.task == 'multi_choice_question':
            results = {
                'accuracy':
                multi_choice_question_calculate_accuracy_(
                    predictions, references),
            }
        elif self.task == 'open_question':
            from bert_score import score
            correct_answers = [ref[0] for ref in references]
            my_answers = [pred[0] for pred in predictions]
            P, R, F1 = score(my_answers,
                             correct_answers,
                             lang='en',
                             verbose=False,
                             num_layers=14,
                             model_type='FacebookAI/roberta-large')

            results = {
                # 'bleu': total_bleu/len(my_answers),
                # 'rouge': total_rouge/len(my_answers),
                'bert_score': sum(F1).item() / len(F1),
            }
        else:
            raise ValueError(f'Unknown task: {self.task}')

        return results
