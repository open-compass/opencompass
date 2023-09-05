from collections import defaultdict
from typing import Optional

from mmengine.evaluator import BaseMetric

from opencompass.registry import METRICS


@METRICS.register_module()
class MMEMetric(BaseMetric):
    """Dump model's prediction to a file.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
    """

    task_dict = {
        'Perception': [
            'existence', 'count', 'position', 'color', 'posters', 'celebrity',
            'scene', 'landmark', 'artwork', 'OCR'
        ],
        'Cognition': [
            'commonsense_reasoning', 'numerical_calculation',
            'text_translation', 'code_reasoning'
        ]
    }  # noqa

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)

    def process(self, data_batch, data_samples) -> None:
        for data_sample in data_samples:
            result = dict()
            result['img_path'] = data_sample['img_path']
            result['task'] = data_sample['task']
            result['pred'] = 1 if data_sample['answer'].lower(
            ) == data_sample['pred_answer'].lower() else 0
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:

        # reorganize results
        record = dict()
        for task in (self.task_dict['Perception'] +
                     self.task_dict['Cognition']):
            record[task] = defaultdict(int)
        for sample in results:
            record[sample['task']][sample['img_path']] += sample['pred']

        # compute subtask score
        metric = dict()
        for task in (self.task_dict['Perception'] +
                     self.task_dict['Cognition']):
            single_sum, double_sum = 0., 0.
            for v in record[task].values():
                assert 0 <= v <= 2
                if v == 2:
                    single_sum += 2
                    double_sum += 1
                elif v == 1:
                    single_sum += 1
            acc = single_sum / 2 / len(record[task])
            acc_plus = double_sum / len(record[task])

            metric[task] = {
                'acc': acc,
                'acc_plus': acc_plus,
                'score': 100 * (acc + acc_plus)
            }

        # compute overall score
        score = 0
        for task in self.task_dict['Perception']:
            score += metric[task]['score']
        metric['Perception'] = score

        score = 0
        for task in self.task_dict['Cognition']:
            score += metric[task]['score']
        metric['Cognition'] = score

        metric['Overall'] = metric['Perception'] + metric['Cognition']

        return metric
