import unittest
from unittest.mock import patch

import pandas as pd

from opencompass.datasets.subjective.arena_hard import arenahard_postprocess


class TestArenaHardPostprocess(unittest.TestCase):

    @patch('opencompass.datasets.subjective.arena_hard.'
           'get_judgeanswer_and_reference')
    def test_no_valid_judgements_returns_zero(self, mock_get_results):
        mock_get_results.return_value = (
            [None],
            [dict(answer1='o3-mini-2025-01-31', answer2='candidate')],
        )

        result = arenahard_postprocess({}, 'unused.json')

        self.assertEqual(result['score'], 0)
        self.assertIn('warning', result)

    @patch('opencompass.datasets.subjective.arena_hard.compute_mle_elo')
    @patch('opencompass.datasets.subjective.arena_hard.'
           'get_judgeanswer_and_reference')
    def test_baseline_is_used_for_point_and_bootstrap_scores(
            self, mock_get_results, mock_compute_elo):
        baseline = 'o3-mini-2025-01-31'
        mock_get_results.return_value = (
            ['A>B'],
            [dict(answer1=baseline, answer2='candidate')],
        )
        mock_compute_elo.return_value = pd.Series(
            [1000.0, 1100.0], index=[baseline, 'candidate'])

        result = arenahard_postprocess({},
                                       'unused.json',
                                       base_model_name=baseline)

        self.assertGreater(result['score'], 50)
        self.assertEqual(mock_compute_elo.call_count, 101)
        for call in mock_compute_elo.call_args_list:
            self.assertEqual(call.kwargs['base_model_name'], baseline)


if __name__ == '__main__':
    unittest.main()
