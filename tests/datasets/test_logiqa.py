import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from opencompass.datasets.logiqa import LogiQADataset, _parse, _strip_labels

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

    def _check_zh(self, ds):
        self.assertEqual(set(ds), {'train', 'validation', 'test'})
        ex = ds['test'][0]
        self.assertEqual([ex[o] for o in 'ABCD'],
                         ['一般来说.', '谁如果.', '答题之前.', '由于.'])
        self.assertEqual(ex['correct_option'], 'B')

    def _write_release(self, root):
        for files, text in ((('Train.txt', 'Eval.txt', 'Test.txt'), _EN),
                            (('zh_train.txt', 'zh_eval.txt', 'zh_test.txt'), _ZH)):
            for file_name in files:
                Path(root, file_name).write_text(text, encoding='utf-8')

    @patch('opencompass.datasets.logiqa.get_data_path')
    def test_load_resolved_path(self, mock_get_data_path):
        with TemporaryDirectory() as root:
            self._write_release(root)
            mock_get_data_path.return_value = root
            ds = LogiQADataset.load(path='opencompass/LogiQA', name='zh')
        mock_get_data_path.assert_called_once_with('opencompass/LogiQA')
        self._check_zh(ds)

    def test_load_from_compass_cache(self):
        with TemporaryDirectory() as cache:
            root = Path(cache, 'data', 'LogiQA')
            root.mkdir(parents=True)
            self._write_release(root)
            with patch.dict(os.environ, {
                    'COMPASS_DATA_CACHE': cache,
                    'DATASET_SOURCE': ''
            }):
                ds = LogiQADataset.load(path='opencompass/LogiQA', name='zh')
        self._check_zh(ds)

    def test_load_local_dir(self):
        with TemporaryDirectory() as root:
            self._write_release(root)
            for name in ('en', 'zh'):
                with self.subTest(name=name):
                    ds = LogiQADataset.load(path=root, name=name)
                    if name == 'zh':
                        self._check_zh(ds)
                    else:
                        self.assertEqual(set(ds), {'train', 'validation', 'test'})
                        for split in ds.values():
                            self.assertEqual(len(split), 1)
                            self.assertEqual(split[0]['A'],
                                             '20-year-old accountant.')
                            self.assertEqual(split[0]['correct_option'], 'A')

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            LogiQADataset.load(path='opencompass/LogiQA', name='fr')


if __name__ == '__main__':
    unittest.main()
