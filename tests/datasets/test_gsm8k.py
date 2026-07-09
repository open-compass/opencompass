import unittest

from opencompass.datasets.gsm8k import (Gsm8kEvaluator,
                                        gsm8k_dataset_postprocess,
                                        gsm8k_postprocess)


class TestGSM8KPostprocess(unittest.TestCase):

    def test_extracts_comma_separated_final_answer(self):
        pred = gsm8k_postprocess('The answer is $70,000.')
        answer = gsm8k_dataset_postprocess('#### 70000')
        score = Gsm8kEvaluator().score([pred], [answer])

        self.assertEqual(pred, '70000')
        self.assertTrue(score['details'][0]['correct'])

    def test_extracts_negative_decimal(self):
        self.assertEqual(gsm8k_postprocess('The answer is -3.75.'),
                         '-3.75')

    def test_ignores_following_question(self):
        text = 'The answer is 42.\nQuestion: another example uses 100.'

        self.assertEqual(gsm8k_postprocess(text), '42')


if __name__ == '__main__':
    unittest.main()
