import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from opencompass.datasets.logiqa import (_FILES, LogiQADataset, _parse,
                                         _strip_labels)

# Blocks taken from the original files. The first one is mis-parsed by the
# HF mirror scripts, which turn '20-year-old' into '0-year-old'.
_EN = ('\n'
       'a\n'
       'Context one.\n'
       'Which is true?\n'
       'A.20-year-old accountant\n'
       'B.24-year-old salesperson\n'
       'C.A.B and C are players\n'
       'D.Ding\n'
       '\n'
       'c\n'
       'Context two.\n'
       'Which is false?\n'
       'A No dangshen\n'
       'B No Shouwu\n'
       'C No baizhu\n'
       'D Has baizhu\n')
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

    def test_space_label_only_when_all_options_use_it(self):
        self.assertEqual(
            _strip_labels(['A No x', 'B No y', 'C 有白术', 'D 没有白术']),
            ['No x', 'No y', '有白术', '没有白术'])
        self.assertEqual(_strip_labels(['A man came', 'B.x', 'C.y', 'D.z']),
                         ['A man came', 'x', 'y', 'z'])

    def test_keeps_mismatched_label(self):
        self.assertEqual(_strip_labels(['A.x', 'C.Dell: y', 'B.Dow z', 'D.w']),
                         ['x', 'C.Dell: y', 'B.Dow z', 'w'])

    def test_keeps_names_after_label(self):
        self.assertEqual(
            _strip_labels(['A.A.B, C', 'B.B, C', 'C.C.D', 'D.Ding']),
            ['A.B, C', 'B, C', 'C.D', 'Ding'])


class TestLogiQADataset(unittest.TestCase):

    def _write(self, root, name, text):
        for file_name, _ in _FILES[name].values():
            with open(f'{root}/{file_name}', 'w', encoding='utf-8') as f:
                f.write(text)

    def test_parse(self):
        with TemporaryDirectory() as root:
            self._write(root, 'en', _EN)
            data = _parse(f'{root}/Test.txt')
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['A'], '20-year-old accountant')
        self.assertEqual(data[0]['C'], 'A.B and C are players')
        self.assertEqual(data[0]['correct_option'], 'A')
        self.assertEqual(data[1]['A'], 'No dangshen')
        self.assertEqual(data[1]['correct_option'], 'C')

    def test_load_local_files(self):
        with TemporaryDirectory() as root:
            self._write(root, 'zh', _ZH)
            ds = LogiQADataset.load(path=root, name='zh')
        self.assertEqual(set(ds), {'train', 'validation', 'test'})
        ex = ds['test'][0]
        self.assertEqual([ex[o] for o in 'ABCD'],
                         ['一般来说', '谁如果', '答题之前', '由于'])
        self.assertEqual(ex['correct_option'], 'B')

    def test_legacy_hf_id_maps_to_original_files(self):
        with TemporaryDirectory() as root:
            self._write(root, 'en', _EN)
            with patch('opencompass.datasets.logiqa._DEFAULT_ROOT', root):
                ds = LogiQADataset.load(path='lucasmccabe/logiqa',
                                        trust_remote_code=True)
        self.assertEqual(ds['test'][0]['A'], '20-year-old accountant')

    def test_malformed_file_raises(self):
        with TemporaryDirectory() as root:
            self._write(root, 'en', _EN.replace('\na\n', '\nx\n', 1))
            with self.assertRaises(ValueError):
                _parse(f'{root}/Test.txt')


if __name__ == '__main__':
    unittest.main()
