from datasets import Dataset, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class LongBenchv2Dataset(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path)
        dataset = load_dataset('json', data_files=path)

        split = 'train'
        raw_data = []
        for i in range(len(dataset[split])):
            question = dataset[split]['question'][i]
            context = dataset[split]['context'][i]
            answer = dataset[split]['answer'][i]
            choice_A = dataset[split]['choice_A'][i]
            choice_B = dataset[split]['choice_B'][i]
            choice_C = dataset[split]['choice_C'][i]
            choice_D = dataset[split]['choice_D'][i]
            difficulty = dataset[split]['difficulty'][i]
            length = dataset[split]['length'][i]
            raw_data.append({
                'question': question,
                'context': context,
                'answer': answer,
                'choice_A': choice_A,
                'choice_B': choice_B,
                'choice_C': choice_C,
                'choice_D': choice_D,
                'difficulty': difficulty,
                'length': length
            })
        dataset['test'] = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module()
class LongBenchv2Evaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def score(self, predictions, references, test_set):
        if not test_set:
            raise ValueError('test set is empty')

        metrics = {
            'total': {
                'correct': 0,
                'total': 0
            },
            'difficulty': {
                'easy': {
                    'correct': 0,
                    'total': 0
                },
                'hard': {
                    'correct': 0,
                    'total': 0
                }
            },
            'length': {
                'short': {
                    'correct': 0,
                    'total': 0
                },
                'medium': {
                    'correct': 0,
                    'total': 0
                },
                'long': {
                    'correct': 0,
                    'total': 0
                }
            }
        }

        for i, (pred, ref,
                sample) in enumerate(zip(predictions, references, test_set)):
            is_correct = (pred == ref)

            metrics['total']['total'] += 1
            if is_correct:
                metrics['total']['correct'] += 1

            difficulty = sample.get('difficulty', 'unknown')
            if difficulty in metrics['difficulty']:
                metrics['difficulty'][difficulty]['total'] += 1
                if is_correct:
                    metrics['difficulty'][difficulty]['correct'] += 1

            length = sample.get('length', 'unknown')
            if length in metrics['length']:
                metrics['length'][length]['total'] += 1
                if is_correct:
                    metrics['length'][length]['correct'] += 1

        results = {
            'accuracy':
            metrics['total']['correct'] / metrics['total']['total'] * 100
        }

        for diff in ['easy', 'hard']:
            if metrics['difficulty'][diff]['total'] > 0:
                acc = metrics['difficulty'][diff]['correct'] / metrics[
                    'difficulty'][diff]['total'] * 100
                results[f'accuracy_{diff}'] = acc

        for length in ['short', 'medium', 'long']:
            if metrics['length'][length]['total'] > 0:
                acc = metrics['length'][length]['correct'] / metrics['length'][
                    length]['total'] * 100
                results[f'accuracy_{length}'] = acc

        return results
