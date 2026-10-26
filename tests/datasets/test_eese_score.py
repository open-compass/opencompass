import unittest

from opencompass.datasets.eese.eese_postprocessors import \
    eese_score_postprocess_dict
from opencompass.datasets.eese.utils import extract_first_numeric_score


class TestEESEScore(unittest.TestCase):

    def test_a_word_judgment_of_correct_is_not_scored_as_zero(self):
        self.assertIsNone(extract_first_numeric_score('correct'))
        result = eese_score_postprocess_dict({'0': {
            'prediction': 'correct'
        }}, '')
        self.assertEqual(result['details']['0']['score'], 10)
        self.assertEqual(result['overall_score'], 100)

        incorrect = eese_score_postprocess_dict(
            {'0': {
                'prediction': 'The answer is incorrect.'
            }}, '')
        self.assertEqual(incorrect['details']['0']['score'], 0)
        self.assertEqual(incorrect['overall_score'], 0)

    def test_a_digit_score_is_unchanged(self):
        self.assertEqual(extract_first_numeric_score('8'), 8)
        result = eese_score_postprocess_dict({'0': {
            'prediction': 'score: 6'
        }}, '')
        self.assertEqual(result['details']['0']['score'], 6)
        self.assertEqual(result['overall_score'], 60)


if __name__ == '__main__':
    unittest.main()
