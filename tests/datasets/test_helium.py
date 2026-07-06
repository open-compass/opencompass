import json
import unittest
from unittest.mock import patch

from datasets import Dataset

from opencompass.datasets.helium import (
    HeliumMarketResolutionDataset,
    HeliumMarketResolutionEvaluator,
    score_market_resolution_item,
)


class TestHeliumMarketResolution(unittest.TestCase):

    def test_iv_score_accepts_decimal_percent(self):
        item = {
            'task': 'implied_volatility',
            'ground_truth': {
                'iv_percent': 50.0,
            },
            'scoring_tier': 'core',
        }

        self.assertEqual(score_market_resolution_item(item, '0.5'), 1.0)
        self.assertAlmostEqual(
            score_market_resolution_item(item, '59'),
            1.0 - 9.0 / 18.0,
        )

    def test_mcq_score_parses_first_line_letter(self):
        item = {
            'task': 'relative_iv',
            'ground_truth': {
                'answer': 'B',
            },
            'scoring_tier': 'core',
        }

        self.assertEqual(
            score_market_resolution_item(item, 'B) higher IV'), 1.0)
        self.assertEqual(score_market_resolution_item(item, 'A'), 0.0)

    def test_evaluator_uses_core_score_as_primary_score(self):
        evaluator = HeliumMarketResolutionEvaluator()
        references = [
            json.dumps({
                'task': 'relative_iv',
                'ground_truth': {
                    'answer': 'A',
                },
                'scoring_tier': 'core',
            }),
            json.dumps({
                'task': 'canary_watermark',
                'ground_truth': {},
                'scoring_tier': 'diagnostic',
            }),
        ]

        result = evaluator.score(['B', 'UNKNOWN'], references)

        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['core_score'], 0.0)
        self.assertEqual(result['overall_score'], 50.0)

    @patch('opencompass.datasets.helium.load_dataset')
    def test_dataset_load_builds_reference_column(self, mock_load_dataset):
        mock_load_dataset.return_value = Dataset.from_list([
            {
                'task': 'relative_iv',
                'prompt': 'Reply A or B.',
                'ground_truth': '{"answer":"A"}',
                'scoring_tier': 'core',
            }
        ])

        loaded = HeliumMarketResolutionDataset.load('test_path')
        item = loaded['test'][0]
        reference = json.loads(item['reference'])

        mock_load_dataset.assert_called_once_with('test_path', split='test')
        self.assertEqual(reference['task'], 'relative_iv')
        self.assertEqual(reference['ground_truth'], {'answer': 'A'})
        self.assertEqual(reference['scoring_tier'], 'core')


if __name__ == '__main__':
    unittest.main()
