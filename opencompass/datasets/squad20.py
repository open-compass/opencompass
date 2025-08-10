import json

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.utils import get_data_path
from opencompass.utils.text_postprocessors import general_postprocess

from .base import BaseDataset


class SQuAD20Dataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        with open(path, 'r') as f:
            data = json.load(f)
        data = data['data']
        dataset = []
        for article in data:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    is_impossible = qa['is_impossible']
                    if not is_impossible:
                        answers = list(
                            set([answer['text'] for answer in qa['answers']]))
                    else:
                        answers = list(
                            set([
                                answer['text']
                                for answer in qa['plausible_answers']
                            ]))
                        answers += ['impossible to answer']
                    item = {
                        'context': paragraph['context'],
                        'question': qa['question'],
                        'answers': answers,
                    }
                    dataset.append(item)
        dataset = Dataset.from_list(dataset)
        return dataset


class SQuAD20Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        processed_predictions = []
        for prediction in predictions:
            prediction = prediction.split('\n')[0].lower()
            if 'answer is' in prediction:
                prediction = prediction.split('answer is')[-1]
            prediction = general_postprocess(prediction)
            processed_predictions.append(prediction)
        processed_answers = [[general_postprocess(j).lower() for j in i]
                             for i in references]

        cnt = 0
        for pred, cand_ans in zip(processed_predictions, processed_answers):
            cnt += int(any([cand == pred for cand in cand_ans]))
        score = cnt / len(predictions) * 100

        return {'score': score}
