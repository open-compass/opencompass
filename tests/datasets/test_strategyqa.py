import unittest

from opencompass.datasets.strategyqa import strategyqa_pred_postprocess

MULTI_PARAGRAPH = (
    'Genghis Khan and his descendants had a large number of children, which '
    'expanded his lineage. Julius Caesar had fewer known descendants.\n\n'
    'Estimates put his direct descendants at around 16 million.\n\n'
    'So the answer is yes')


class TestStrategyqaPredPostprocess(unittest.TestCase):

    def test_reads_the_verdict_from_the_last_paragraph(self):
        self.assertEqual(strategyqa_pred_postprocess(MULTI_PARAGRAPH), 'yes')

    def test_does_not_match_no_inside_a_word(self):
        self.assertEqual(strategyqa_pred_postprocess('Hamsters are not prey.'),
                         '')

    def test_reads_a_single_paragraph_verdict(self):
        self.assertEqual(strategyqa_pred_postprocess('The answer is no.'),
                         'no')

    def test_keeps_the_verdict_when_meta_text_follows(self):
        text = ('The answer is yes.\n\nThe answer is not obvious; '
                'it could be no.')
        self.assertEqual(strategyqa_pred_postprocess(text), 'yes')

    def test_uses_the_last_answer_marker(self):
        text = 'The answer is no.\n\nThe answer is yes.'
        self.assertEqual(strategyqa_pred_postprocess(text), 'yes')

    def test_does_not_let_reasoning_override_the_verdict(self):
        self.assertEqual(
            strategyqa_pred_postprocess('Yes. There is no doubt.'), 'yes')

    def test_reads_a_verdict_without_the_marker(self):
        text = 'Cats cannot fly.\n\nSo yes, they stay grounded.'
        self.assertEqual(strategyqa_pred_postprocess(text), 'yes')


if __name__ == '__main__':
    unittest.main()
