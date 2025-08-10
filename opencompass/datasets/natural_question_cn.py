import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.utils import get_data_path
from opencompass.utils.text_postprocessors import general_postprocess

from .base import BaseDataset


class NaturalQuestionDatasetCN(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            filename = osp.join(path, f'{split}.jsonl')
            all_data = []
            with open(filename, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if split == 'dev':
                        data['answer'] = data['answer'][0]
                    all_data.append(data)
                dataset[split] = Dataset.from_list(all_data)

        return dataset


class NQEvaluatorCN(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        processed_predictions = []
        for prediction in predictions:
            prediction = prediction.split('\n')[0].lower()
            if '答案是：' in prediction:
                prediction = prediction.split('答案是：')[-1]
            prediction = general_postprocess(prediction)
            processed_predictions.append(prediction)
        processed_answers = [[general_postprocess(j).lower() for j in i]
                             for i in references]

        cnt = 0
        for pred, cand_ans in zip(processed_predictions, processed_answers):
            cnt += int(any([cand == pred for cand in cand_ans]))
        score = cnt / len(predictions) * 100

        return {'score': score}
