import json
import os.path as osp
import os
from typing import List

from datasets import Dataset
import evaluate

from opencompass.openicl.icl_evaluator import HuggingfaceEvaluator
from opencompass.utils import get_data_path

from .base import BaseDataset


os.environ["HF_ALLOW_CODE_EVAL"] = "1"


class HumanevalevalProDataset(BaseDataset):

    @staticmethod
    def load(path, num_repeats=1, local_mode=False):
        path = get_data_path(path, local_mode=local_mode)
        dataset = []
        with open(path, encoding='utf-8') as f:
            raw_data = json.load(f)
            for data in raw_data:
                dataset.extend(
                        [data for _ in range(num_repeats)])
        return Dataset.from_list(dataset)
    

class HumanevalProEvaluator(HuggingfaceEvaluator):

    def _preprocess(self, predictions, references):
        predictions = [[_] for _ in predictions]
        return {
            'predictions': predictions,
            'references': references,
        }
    
    def _postprocess(self, scores):
        scores = {f'humaneval_{k}': scores[k] * 100 for k in scores}
        return scores

    def score(self, predictions, references, test_set):
        # predictions are LLM's output; references are the 'output_column' of 'humanevalpro_reader_cfg'
        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }

        # use codes pre-downloaded to opencompass repo, avoid downloading
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parrent_dir = os.path.dirname(current_dir)
        local_path = os.path.join(parrent_dir, 'openicl', 'icl_evaluator', 'hf_metrics', self.metric)

        if os.path.exists(local_path):
            metric = evaluate.load(local_path)
        else:
            metric = evaluate.load(self.metric)
        scores, _ = metric.compute(**self._preprocess(predictions, references), 
                                k=[1, 3, 5], 
                                num_workers=4)
        result = self._postprocess(scores)
        return result


def humanevalpro_postprocess_official(text):
    """
    The official post-processing method for humaneval_pro, which is solely applicable to the complete generation paradigm. 
    The chat template paradigm requires a different post-processing method.
    """
    text = text[: index if (index := text.find("```")) != -1 else len(text)]
    return text


def humanevalpro_postprocess_oc(text):
    """
    For those generated based on the chat template paradigm, this method is recommended.
    """
    start = text.rfind("```python") + len("```python")
    end = text.find("```", start)

    code = text[start:end].strip()
    return code
