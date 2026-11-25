import unittest

from opencompass.datasets.MedXpertQA import answer_cleansing


class TestMedXpertQAAnswerCleansing(unittest.TestCase):

    def setUp(self):
        self.options = ['opt A', 'opt B', 'opt C', 'opt D']

    def test_prediction_without_option_letter_returns_empty(self):
        # A prediction that contains no bare option letter (a refusal, an
        # empty/blank generation, or a non-Latin answer) must return the empty
        # sentinel, not raise IndexError and abort scoring of the whole
        # benchmark. Mirrors the sibling Medbullets.answer_cleansing guard.
        for pred in ['I cannot answer this question.', '', '答案无法确定',
                     'The correct choice is unclear']:
            with self.subTest(pred=pred):
                self.assertEqual(
                    answer_cleansing('zero-shot', pred, self.options, 'C'), [])

    def test_prediction_with_option_letter_is_extracted(self):
        self.assertEqual(
            answer_cleansing('zero-shot', 'The answer is C.', self.options, 'C'),
            'C')
        self.assertEqual(
            answer_cleansing('zero-shot', 'B is correct', self.options, 'B'),
            'B')


if __name__ == '__main__':
    unittest.main()
