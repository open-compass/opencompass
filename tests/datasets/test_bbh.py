import unittest

from opencompass.datasets.bbh import BBHEvaluator, bbh_freeform_postprocess


class TestBBHFreeform(unittest.TestCase):

    def test_freeform_postprocess_extracts_short_answers(self):
        self.assertEqual(bbh_freeform_postprocess('no'), 'No')
        self.assertEqual(
            bbh_freeform_postprocess(
                'So the answer is No, because the last speaker lies.'),
            'No',
        )
        self.assertEqual(bbh_freeform_postprocess('Answer is **yes**.'),
                         'Yes')

    def test_freeform_postprocess_normalizes_word_sorting_commas(self):
        self.assertEqual(
            bbh_freeform_postprocess('apple, banana, cherry'),
            'apple banana cherry',
        )
        self.assertEqual(bbh_freeform_postprocess('1,000'), '1,000')

    def test_freeform_evaluator_accepts_equivalent_formats(self):
        result = BBHEvaluator().score(
            predictions=[
                'no',
                'No, because the last speaker lies.',
                'apple, banana, cherry',
            ],
            references=[
                'No',
                'No',
                'apple banana cherry',
            ],
        )

        self.assertEqual(result['score'], 100.0)
        self.assertTrue(all(item['correct'] for item in result['details']))

    def test_freeform_evaluator_rejects_wrong_word_order(self):
        result = BBHEvaluator().score(
            predictions=['apple, cherry, banana'],
            references=['apple banana cherry'],
        )

        self.assertEqual(result['score'], 0.0)
        self.assertFalse(result['details'][0]['correct'])


if __name__ == '__main__':
    unittest.main()
