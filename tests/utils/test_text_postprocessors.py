"""Unit tests for text_postprocessors."""

import unittest
from unittest.mock import patch

from opencompass.utils import text_postprocessors as tp


class TestTextPostprocessors(unittest.TestCase):
    """Test cases for text_postprocessors."""

    def test_general_postprocess(self):
        text = 'The quick brown'
        self.assertEqual(tp.general_postprocess(text), 'quick brown')

    @patch('jieba.cut')
    def test_general_cn_postprocess(self, mock_cut):
        mock_cut.return_value = ['今', '天', '天', '气', '好']
        text = '今天天气好'
        self.assertEqual(tp.general_cn_postprocess(text), '今 天 天 气 好')
        mock_cut.assert_called_with(text)

    def test_first_capital_postprocess(self):
        self.assertEqual(tp.first_capital_postprocess('abCDef'), 'C')
        self.assertEqual(tp.first_capital_postprocess('abcdef'), '')

    def test_last_capital_postprocess(self):
        self.assertEqual(tp.last_capital_postprocess('abCDef'), 'D')
        self.assertEqual(tp.last_capital_postprocess('abcdef'), '')

    def test_think_pred_postprocess(self):
        text = 'answer: B'
        pattern = r'answer:\s*([A-D])'
        self.assertEqual(tp.think_pred_postprocess(text, pattern), 'B')
        self.assertEqual(tp.think_pred_postprocess('no match', pattern),
                         'no match')

    def test_first_option_postprocess(self):
        text = '答案是 B'
        self.assertEqual(tp.first_option_postprocess(text, 'ABCD'), 'B')

    def test_last_option_postprocess(self):
        text = 'A then C then B'
        self.assertEqual(tp.last_option_postprocess(text, 'ABC'), 'B')

    def test_first_number_postprocess(self):
        self.assertEqual(tp.first_number_postprocess('val=3.14; x'), 3.14)
        self.assertEqual(tp.first_number_postprocess('-42 is here'), -42.0)

    def test_multiple_select_postprocess(self):
        text = 'bA D C'
        self.assertEqual(tp.multiple_select_postprocess(text), 'ACD')

    def test_xml_tag_postprocessor(self):
        text = '<conclude>first</conclude> noise <conclude>last</conclude>'
        self.assertEqual(tp.xml_tag_postprocessor(text, '<conclude>'), 'last')

    def test_general_eval_wrapper_postprocess(self):
        text = "'The fox'"
        self.assertEqual(
            tp.general_eval_wrapper_postprocess(text, postprocess='general'),
            'fox')

    def test_match_answer_pattern(self):
        text = 'Final answer: 123'
        self.assertEqual(tp.match_answer_pattern(text, r'answer: (\d+)'),
                         '123')

    def test_extract_non_reasoning_content(self):
        text = 'This is a test.</think> How are you?'
        self.assertEqual(tp.extract_non_reasoning_content(text),
                         'How are you?')
        text = 'Start<think>reasoning</think> End'
        self.assertEqual(tp.extract_non_reasoning_content(text), 'Start End')


if __name__ == '__main__':
    unittest.main()
