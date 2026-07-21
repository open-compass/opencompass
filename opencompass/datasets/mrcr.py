import difflib
import json
import re

from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


@LOAD_DATASET.register_module()
class MRCRDataset(BaseDataset):

    @staticmethod
    def load(path='giulio98/MRCR_v2_common',
             subset='2needle_in_4096_8192',
             **kwargs):
        dataset = load_dataset(path, subset, split='test')
        data = []
        for sample in dataset:
            # Extract the random hash prefix from the question
            m = re.search(r'Prepend\s+(\S+)\s+to', sample['question'])
            hash_prefix = m.group(1) if m else ''
            gold = json.dumps({
                'answer': sample['answers'][0],
                'prefix': hash_prefix,
            })
            data.append({
                'context': sample['context'],
                'question': sample['question'],
                'gold': gold,
            })
        return Dataset.from_list(data)


@ICL_EVALUATORS.register_module()
class MRCREvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        details = []
        scores = []
        for pred, ref in zip(predictions, references):
            gold = json.loads(ref)
            answer = gold['answer']
            prefix = gold['prefix']

            if not pred.startswith(prefix):
                scores.append(0.0)
                details.append({
                    'pred': pred,
                    'answer': answer,
                    'score': 0.0,
                })
                continue

            stripped_pred = pred[len(prefix):]
            stripped_answer = answer[len(prefix):]
            ratio = difflib.SequenceMatcher(None, stripped_pred,
                                            stripped_answer).ratio()
            scores.append(ratio)
            details.append({
                'pred': pred,
                'answer': answer,
                'score': ratio,
            })

        avg_score = sum(scores) / len(scores) * 100
        return {'score': avg_score, 'details': details}
