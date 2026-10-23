import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from opencompass.datasets.logiqa import (_HF_SCRIPTS, LogiQADataset, _parse,
                                         _strip_labels)

# Blocks in the format of the original LogiQA txt files. The HF mirror scripts
# turn '20-year-old' into '0-year-old' and '一般来说' into '般来说'.
_EN = ('\n'
       'a\n'
       'Context one.\n'
       'Which is true?\n'
       'A.20-year-old accountant\n'
       'B. 24-year-old salesperson\n'
       'C.A.B and C are players\n'
       'D.Ding\n')
_ZH = ('\n'
       'b\n'
       '上下文。\n'
       '问题？\n'
       'A．一般来说\n'
       'B,谁如果\n'
       'C：答题之前\n'
       'D、由于\n')


class TestLogiQAStripLabels(unittest.TestCase):

    def test_keeps_first_character(self):
        self.assertEqual(
            _strip_labels(['A.20-year-old', 'B．一般', 'C,foo', 'D：bar']),
            ['20-year-old', '一般', 'foo', 'bar'])

    def test_same_as_original_for_labels_with_space(self):
        # Matches the original `answer[3:]` when it was correct.
        self.assertEqual(_strip_labels(['A. x', 'B. y', 'C : z', 'D. w']),
                         ['x', 'y', ' z', 'w'])

    def test_space_label_only_when_all_options_use_it(self):
        self.assertEqual(
            _strip_labels(['A No x', 'B No y', 'C 有白术', 'D 没有白术']),
            ['No x', 'No y', '有白术', '没有白术'])
        self.assertEqual(_strip_labels(['A man came', 'B.x', 'C.y', 'D.z']),
                         ['A man came', 'x', 'y', 'z'])

    def test_keeps_names_after_label(self):
        self.assertEqual(
            _strip_labels(['A.A.B, C', 'B.B, C', 'C.C.D', 'D.Ding']),
            ['A.B, C', 'B, C', 'C.D', 'Ding'])


class TestLogiQADataset(unittest.TestCase):

    def _write(self, root, text):
        path = f'{root}/data.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return path

    def test_parse(self):
        with TemporaryDirectory() as root:
            data = _parse(self._write(root, _EN))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['A'], '20-year-old accountant.')
        self.assertEqual(data[0]['B'], '24-year-old salesperson.')
        # `_process_sentences` adds a space after each '.', as upstream does.
        self.assertEqual(data[0]['C'], 'A. B and C are players.')
        self.assertEqual(data[0]['correct_option'], 'A')

    @patch('opencompass.datasets.logiqa.DownloadManager')
    def test_load_hf_id_reads_original_files(self, mock_dm):
        with TemporaryDirectory() as root:
            path = self._write(root, _ZH)
            mock_dm.return_value.download_and_extract.side_effect = \
                lambda urls: {split: path for split in urls}
            ds = LogiQADataset.load(path='jiacheng-ye/logiqa-zh',
                                    trust_remote_code=True)
        urls = mock_dm.return_value.download_and_extract.call_args[0][0]
        self.assertEqual(set(urls), set(_HF_SCRIPTS['jiacheng-ye/logiqa-zh']))
        self.assertTrue(urls['test'].endswith('/zh_test.txt'))
        ex = ds['test'][0]
        self.assertEqual([ex[o] for o in 'ABCD'],
                         ['一般来说.', '谁如果.', '答题之前.', '由于.'])
        self.assertEqual(ex['correct_option'], 'B')


if __name__ == '__main__':
    unittest.main()
