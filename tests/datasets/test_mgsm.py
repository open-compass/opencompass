"""Unit tests for the MGSM postprocessor and evaluator."""

import unittest

from opencompass.datasets.mgsm import MGSM_Evaluator, mgsm_postprocess


class TestMGSMPostprocess(unittest.TestCase):

    def test_takes_last_number_after_prefix(self):
        text = '2,000 + 125 = 2,125\nAnswer: 2,125'
        self.assertEqual(mgsm_postprocess(text, 'en'), '2125')

    def test_missing_prefix_returns_empty(self):
        self.assertEqual(mgsm_postprocess('The total is 18.', 'en'), '')


class TestMGSMEvaluator(unittest.TestCase):

    def test_score_ignores_commas_and_trailing_zeros(self):
        # Some references in the MGSM data use thousands separators
        # (e.g. '2,125'); trailing decimal zeros are not a different answer.
        # '180' vs '18' checks that zeros are only stripped after a '.'.
        predictions = ['2125', '18.00', '100', '17', '180']
        references = ['2,125', '18', '100', '18', '18']
        result = MGSM_Evaluator().score(predictions, references)
        self.assertEqual(
            [result['details'][str(i)]['correct'] for i in range(5)],
            [True, True, True, False, False])
        self.assertEqual(result['accuracy'], 60.0)

    def test_postprocessed_answer_matches_comma_reference(self):
        prediction = mgsm_postprocess('So she pays $2,125.\nAnswer: 2,125',
                                      'en')
        result = MGSM_Evaluator().score([prediction], ['2,125'])
        self.assertEqual(result['accuracy'], 100.0)


if __name__ == '__main__':
    unittest.main()
