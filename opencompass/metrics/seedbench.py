import torch
from mmengine.evaluator import BaseMetric

from opencompass.registry import METRICS

EVAL_DIM_MAPPING = {
    1: 'Scene Understanding',
    2: 'Instance Identity',
    3: 'Instance Attributes',
    4: 'Instance Location',
    5: 'Instance Counting',
    6: 'Spatial Relations',
    7: 'Instance Interaction',
    8: 'Visual Reasoning',
    9: 'Text Recognition',
    10: 'Action Recognition',
    11: 'Action Prediction',
    12: 'Procedure Understanding',
}


@METRICS.register_module()
class SEEDBenchAcc(BaseMetric):
    """Compute results for SEED-Bench."""

    def process(self, data_batch, data_samples) -> None:
        for data_sample in data_samples:
            losses = data_sample['losses']
            class_ranks = torch.argsort(losses, dim=-1).cpu()
            pred_id = ['A', 'B', 'C', 'D'][class_ranks[0]]
            answer_record = {
                'q_id': data_sample['question_id'],
                'prediction': pred_id,
                'gt': data_sample['answer'],
                'q_type_id': data_sample['question_type_id'],
                'losses': [str(num) for num in list(losses.cpu().numpy())],
            }
            self.results.append(answer_record)

    def compute_metrics(self, results: list) -> dict:
        type_counts = {}
        correct_counts = {}
        out = {}
        out['answer_records'] = results
        for item in results:
            pred, gt = item['prediction'], item['gt']
            data_type = item['q_type_id']

            type_counts[data_type] = type_counts.get(data_type, 0) + 1
            if pred == gt:
                correct_counts[data_type] = correct_counts.get(data_type,
                                                               0) + 1

        total_count = 0
        total_correct = 0
        for data_type in type_counts.keys():
            accuracy = correct_counts.get(data_type,
                                          0) / type_counts[data_type] * 100
            category = EVAL_DIM_MAPPING[data_type]
            out[f'Data type {data_type} - {category}'] = accuracy

            total_count += type_counts[data_type]
            total_correct += correct_counts.get(data_type, 0)

        total_accuracy = total_correct / total_count * 100
        out['Total accuracy'] = total_accuracy
        return out
