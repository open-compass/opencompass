import unittest

from opencompass.datasets.eese.eese_postprocessors import \
    eese_score_postprocess_dict
from opencompass.datasets.eese.utils import extract_first_numeric_score


class TestExtractFirstNumericScore(unittest.TestCase):

    def test_score_is_read_from_surrounding_text(self):
        self.assertEqual(extract_first_numeric_score('Score: 7'), 7)

    def test_text_without_digits_has_no_score(self):
        self.assertIsNone(extract_first_numeric_score('Correct'))


class TestEESEScorePostprocess(unittest.TestCase):

    def score_of(self, prediction):
        output = {'q1': {'prediction': prediction}}
        result = eese_score_postprocess_dict(output, '')
        return result['details']['q1']['score']

    def test_worded_verdicts_are_graded_by_keyword(self):
        self.assertEqual(self.score_of('Correct'), 10)
        self.assertEqual(self.score_of('Incorrect'), 0)
        self.assertEqual(self.score_of('no verdict given'), 0)


if __name__ == '__main__':
    unittest.main()
