import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path
from opencompass.utils.text_postprocessors import general_postprocess

from .base import BaseDataset


@LOAD_DATASET.register_module()
class NaturalQuestionDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path)
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            if environ.get('DATASET_SOURCE') == 'ModelScope':
                from modelscope.msdatasets import MsDataset
                ms_dataset = MsDataset.load(path, split=split)
                raw_data = []
                for row in ms_dataset:
                    question = row['question']
                    answers = eval(row['answer'])
                    if split == 'dev':
                        answers = answers[0]
                    raw_data.append({'question': question, 'answer': answers})
            else:
                filename = osp.join(path, f'nq-{split}.qa.csv')
                with open(filename, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t')
                    raw_data = []
                    for row in reader:
                        assert len(row) == 2
                        question = row[0]
                        answers = eval(row[1])
                        if split == 'dev':
                            answers = answers[0]
                        raw_data.append({
                            'question': question,
                            'answer': answers
                        })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset


@LOAD_DATASET.register_module()
class NQOpenDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        dataset = DatasetDict()
        for split in ['validation', 'train']:
            filename = osp.join(path, f'nq-open-{split}.jsonl')
            raw_data = []
            with open(filename, 'r', encoding='utf-8') as f:
                for doc in f:
                    doc = json.loads(doc)
                    if split == 'train':
                        doc['answer'] = doc['answer'][0]
                    raw_data.append(doc)
            dataset[split] = Dataset.from_list(raw_data)

        return dataset


@ICL_EVALUATORS.register_module()
class NQEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        processed_predictions = []
        for prediction in predictions:
            prediction = prediction.strip().split('\n')[0].lower()
            if 'answer is' in prediction:
                prediction = prediction.split('answer is')[-1]
            prediction = general_postprocess(prediction)
            processed_predictions.append(prediction)
        processed_answers = [[general_postprocess(j).lower() for j in i]
                             for i in references]

        details = []
        cnt = 0
        for pred, cand_ans in zip(processed_predictions, processed_answers):
            detail = {'pred': pred, 'answer': cand_ans, 'correct': False}
            # is_correct = any([cand == pred for cand in cand_ans])
            is_correct = any([cand in pred for cand in cand_ans])
            cnt += int(is_correct)
            detail['correct'] = is_correct
            details.append(detail)
        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details}
