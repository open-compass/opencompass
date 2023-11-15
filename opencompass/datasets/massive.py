import os
import re
import json
from .base import BaseDataset
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.utils.text_postprocessors import general_postprocess

@LOAD_DATASET.register_module()
class MassiveDataset(BaseDataset):
    @staticmethod
    def load(path):
        dataset = DatasetDict()
        for file in os.listdir(path):
            if '.jsonl' not in file: continue
            split = file.split('.')[0]
            data_list = []
            origin_data = [json.loads(line.strip()) for line in open(os.path.join(path, file))]
            for item in origin_data:
                output = re.findall(r'\[.*?\]', item['target'])
                data_list.append({'text': item['input'], 'output': "".join([p.replace('[', '').replace(']', ' ; ') for p in output])} )
            dataset[split] = Dataset.from_list(data_list)
        return dataset
    

@ICL_EVALUATORS.register_module()
class MassiveEvaluator(BaseEvaluator):
    def f1_score(self, prediction, ground_truth):
        prediction_tokens = general_postprocess(prediction).split(" ; ")
        ground_truth_tokens = general_postprocess(ground_truth).split(" ; ")
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def score(self, predictions, references):
        f1 = total = 0
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        for prediction, reference in zip(predictions, references):
            prediction = prediction.split('\n')[0].lower()
            f1 += f1_score(prediction, reference)
            total += 1

        f1 = 100.0 * f1 / total

        return {'score': f1}


    
