"""Unit tests for the BBEH postprocessors and evaluators."""

import unittest

from opencompass.datasets.bbeh import (BBEHEvaluator, BBEHEvaluator_mcq,
                                       bbeh_freeform_postprocess,
                                       bbeh_mcq_postprocess)


def _correct(result):
    return [d['correct'] for d in result['details']]


class TestBBEHFreeform(unittest.TestCase):

    def test_official_examples(self):
        # The first four examples at the end of google-deepmind/bbeh
        # bbeh/evaluate.py, with the results the official code prints.
        predictions = [
            'Ok The final answer is: \\boxed{4}.',
            '[Reasoning] The final answer is: \\boxed{4}.',
            'Alright! The final answer is: 2, 3, 4',
            'blah blah The final answer is: 2, 3, 4',
        ]
        references = ['4', '3', '2,3,4', '2,3,5']
        result = BBEHEvaluator().score(predictions, references)
        self.assertEqual(_correct(result), [True, False, True, False])

    def test_trailing_period_is_dropped(self):
        self.assertEqual(bbeh_freeform_postprocess('The answer is: 42.'), '42')
        self.assertEqual(bbeh_freeform_postprocess('The answer is: valid.'),
                         'valid')
        predictions = [
            'So the total is 42. The answer is: 42.',
            'The answer is: valid.',
            'The answer is: 41.',
        ]
        result = BBEHEvaluator().score(predictions, ['42', 'valid', '42'])
        self.assertEqual(_correct(result), [True, True, False])
        self.assertAlmostEqual(result['score'], 200 / 3)

    def test_reference_ending_with_period(self):
        # A few references end with a period, e.g. 'I danced.' in
        # bbeh_linguini. They still match once the prediction loses it.
        predictions = ['The answer is: I danced.', 'The answer is: I danced']
        result = BBEHEvaluator().score(predictions, ['I danced.'] * 2)
        self.assertEqual(_correct(result), [True, True])

    def test_with_configured_postprocessors(self):
        # bbeh_gen.py also runs bbeh_mcq_postprocess on the predictions and
        # references of the free-form tasks before BBEHEvaluator scores them.
        # 'sweet (adj.)' is a bbeh_linguini reference.
        predictions = [
            bbeh_mcq_postprocess(p) for p in [
                'The answer is: 42.', 'The answer is: 41.',
                'The answer is: sweet (adj.)'
            ]
        ]
        references = [
            bbeh_mcq_postprocess(r) for r in ['42', '42', 'sweet (adj.)']
        ]
        result = BBEHEvaluator().score(predictions, references)
        self.assertEqual(_correct(result), [True, False, True])


class TestBBEHMultipleChoice(unittest.TestCase):

    def test_trailing_period_and_parentheses(self):
        self.assertEqual(bbeh_mcq_postprocess('So, The answer is: (B).'), 'b')
        self.assertEqual(
            bbeh_mcq_postprocess('The answer is: (C)\nbecause (C) fits.'), 'c')

    def test_official_examples(self):
        # Examples 5 and 6 from the official evaluate.py, plus the
        # trailing-period case.
        predictions = [
            'Ok The answer is: (A)',
            'Ok The answer is: (A)',
            'So, The answer is: (B).',
        ]
        result = BBEHEvaluator_mcq().score(predictions, ['a', 'b', '(B)'])
        self.assertEqual(_correct(result), [True, False, True])


if __name__ == '__main__':
    unittest.main()
