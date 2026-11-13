import builtins
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from opencompass.datasets.cmo_fib import CMOFibDataset
from opencompass.datasets.lawbench.lawbench import LawBenchDataset
from opencompass.datasets.OpenFinData import OpenFinDataDataset
from opencompass.datasets.subjective.alignbench import Config as AlignConfig
from opencompass.datasets.subjective.flames import FlamesDataset

_real_open = builtins.open


def _cp1252_default_open(file, mode='r', *args, **kwargs):
    """Mimic open() on Windows, where text mode defaults to the locale
    encoding (e.g. cp1252 or GBK) instead of UTF-8."""
    if 'b' not in mode and 'encoding' not in kwargs and len(args) < 2:
        kwargs['encoding'] = 'cp1252'
    return _real_open(file, mode, *args, **kwargs)


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False)


def _locale_open(module):
    return patch(f'{module}.open', _cp1252_default_open, create=True)


class TestUTF8DatasetLoaders(unittest.TestCase):
    """The data of these datasets is Chinese text stored as UTF-8, so the
    loaders must not depend on the platform's default encoding."""

    QUESTION = '下列哪一项属于合同法的基本原则？'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, content):
        path = os.path.join(self.dir, name)
        with _real_open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_lawbench(self):
        self._write(
            '1-1.json',
            _dump([{
                'instruction': '请回答',
                'question': self.QUESTION,
                'answer': '诚实信用'
            }]))
        with _locale_open('opencompass.datasets.lawbench.lawbench'):
            ds = LawBenchDataset.load(path=self.dir, index='1-1')
        self.assertEqual(ds[0]['question'], self.QUESTION)

    def test_openfindata(self):
        self._write(
            'emotion_identification.json',
            _dump([{
                'id': 0,
                'question': self.QUESTION,
                'A': '甲',
                'B': '乙',
                'answer': 'A'
            }]))
        with _locale_open('opencompass.datasets.OpenFinData'):
            ds = OpenFinDataDataset.load(path=self.dir,
                                         name='emotion_identification')
        self.assertEqual(ds[0]['question'], self.QUESTION)

    def test_cmo_fib(self):
        path = self._write(
            'cmo.jsonl',
            _dump({
                'origin_prompt': self.QUESTION,
                'gold_answer': '诚实信用'
            }) + '\n')
        with _locale_open('opencompass.datasets.cmo_fib'):
            ds = CMOFibDataset.load(path=path)
        self.assertEqual(ds[0]['question'], self.QUESTION)
        self.assertEqual(ds[0]['answer'], '诚实信用')

    def test_alignbench_config(self):
        self._write(
            'multi-dimension.json',
            _dump({
                'Paths': {
                    'dimension_set_filepath': 'dimension_set.json',
                    'dimension_def_filepath': 'dimension_definition.json',
                    'subcategory_mapping': 'subcategory_question_mapping.json'
                }
            }))
        self._write('dimension_set.json', _dump({'专业能力': ['事实正确性']}))
        self._write('dimension_definition.json',
                    _dump({'事实正确性': '回答中提供的信息是否准确无误'}))
        self._write('subcategory_question_mapping.json',
                    _dump({'专业能力': '专业能力'}))
        with _locale_open('opencompass.datasets.subjective.alignbench'):
            config = AlignConfig(self.dir, 'multi-dimension')
        self.assertEqual(config.category2dimensions('专业能力'), ['事实正确性'])
        self.assertEqual(config.dimension2def('事实正确性'), '回答中提供的信息是否准确无误')

    def test_flames(self):
        self._write('data_protection_config.txt', '请根据以下维度评价回答：\n')
        self._write(
            'data_protection.json',
            _dump([{
                'prompt': self.QUESTION,
                'dimension': '合法性',
                'subcomponent': '数据保护'
            }]))
        with _locale_open('opencompass.datasets.subjective.flames'):
            ds = FlamesDataset(path=self.dir,
                               name='data_protection',
                               reader_cfg=dict(input_columns=['instruction'],
                                               output_column='judge')).test
        self.assertIn(self.QUESTION, ds[0]['instruction'])
        self.assertIn('请根据以下维度评价回答', ds[0]['instruction'])
        self.assertEqual(ds[0]['judge']['dimension'], '合法性')


if __name__ == '__main__':
    unittest.main()
