import os
from typing import Optional

import pandas as pd
from mmengine.evaluator import BaseMetric

from opencompass.registry import METRICS


@METRICS.register_module()
class DumpResults(BaseMetric):
    """Dump model's prediction to a file.

    Args:
        save_path (str): the path to save model's prediction.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
    """

    def __init__(self,
                 save_path: str,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        self.save_path = save_path
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def process(self, data_batch, data_samples) -> None:
        for data_sample in data_samples:
            result = dict()

            result['question'] = data_sample.get('question')
            result.update(data_sample.get('options_dict'))
            result['prediction'] = data_sample.get('pred_answer')
            if data_sample.get('category') is not None:
                result['category'] = data_sample.get('category')
            if data_sample.get('l2-category') is not None:
                result['l2-category'] = data_sample.get('l2-category')
            result['index'] = data_sample.get('index')
            result['split'] = data_sample.get('split')
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        df = pd.DataFrame(results)
        with pd.ExcelWriter(self.save_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        return {}
