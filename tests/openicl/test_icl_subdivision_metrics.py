import unittest

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator


class CorrectnessEvaluator(BaseEvaluator):

    def score(self, predictions):
        return {
            'accuracy': 100 * sum(predictions) / len(predictions),
            'details': [{
                'correct': prediction
            } for prediction in predictions],
        }


class TestSubdivisionMetrics(unittest.TestCase):

    def check_subdivision_metrics(self, subdivisions):
        examples = [{
            'subdivision': subdivision,
            'idx': 0
        } for subdivision in subdivisions]
        result = CorrectnessEvaluator().evaluate(
            k=2,
            n=2,
            original_dataset=Dataset.from_list(examples * 2),
            predictions=[True, False, True, False],
        )

        metrics = [
            f'G-Pass@2_{threshold}'
            for threshold in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        metrics.append('mG-Pass@2')
        for metric in metrics:
            self.assertEqual(result[f'{subdivisions[0]}/{metric}'], 100.0)
            self.assertEqual(result[f'{subdivisions[1]}/{metric}'], 0.0)
            self.assertEqual(result[metric], 50.0)
        self.assertEqual(result['accuracy (2 runs average)'], 50.0)
        self.assertEqual(
            [detail['example_abbr'] for detail in result['details']],
            [f'{subdivision}_0' for subdivision in subdivisions],
        )

    def test_subdivision_names_with_underscores_remain_distinct(self):
        self.check_subdivision_metrics(
            ('math_contest_easy', 'math_contest_hard'))

    def test_subdivision_names_without_underscores(self):
        self.check_subdivision_metrics(('math', 'code'))


if __name__ == '__main__':
    unittest.main()
