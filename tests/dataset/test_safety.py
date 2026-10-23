import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from opencompass.datasets.safety import SafetyDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


class TestSafetyDataset(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.path = Path(self.tmp_dir.name) / 'safety.txt'

    def test_load_returns_test_split_with_contiguous_indices(self):
        self.path.write_text(
            '  first prompt  \n\n\t\n'
            '\u4f60\u597d\uff0c\u4e16\u754c\nfirst prompt\n',
            encoding='utf-8')

        with patch('opencompass.datasets.safety.get_data_path',
                   return_value=str(self.path)) as resolve:
            dataset = SafetyDataset.load('local-safety-prompts')

        resolve.assert_called_once_with('local-safety-prompts',
                                        local_mode=True)
        self.assertIsInstance(dataset, DatasetDict)
        self.assertEqual(list(dataset), ['test'])
        self.assertIsInstance(dataset['test'], Dataset)
        self.assertEqual(dataset['test'].to_list(), [
            dict(idx=0, prompt='first prompt'),
            dict(idx=1, prompt='\u4f60\u597d\uff0c\u4e16\u754c'),
            dict(idx=2, prompt='first prompt'),
        ])

    def test_load_blank_file_returns_empty_test_split(self):
        self.path.write_text('\n \n\t\n', encoding='utf-8')
        with patch('opencompass.datasets.safety.get_data_path',
                   return_value=str(self.path)):
            dataset = SafetyDataset.load('local-safety-prompts')

        self.assertIsInstance(dataset, DatasetDict)
        self.assertEqual(list(dataset), ['test'])
        self.assertEqual(len(dataset['test']), 0)

    def test_load_missing_file_raises(self):
        with patch('opencompass.datasets.safety.get_data_path',
                   return_value=str(self.path)):
            with self.assertRaises(FileNotFoundError):
                SafetyDataset.load('local-safety-prompts')

    def test_dataset_initialization_reader_and_prompt_template(self):
        self.path.write_text(
            '  first prompt  \n\n\u4f60\u597d\uff0c\u4e16\u754c\n',
            encoding='utf-8')
        dataset = SafetyDataset(path=str(self.path),
                                abbr='safety',
                                reader_cfg=dict(input_columns=['prompt'],
                                                output_column='idx',
                                                train_split='test',
                                                test_split='test'))

        self.assertEqual(len(dataset.train), 2)
        self.assertEqual(len(dataset.test), 2)
        self.assertEqual(dataset.test['idx'], [0, 1])
        self.assertEqual(dataset.test['subdivision'], ['safety_test'] * 2)
        self.assertEqual(
            dataset.reader.generate_input_field_corpus(dataset.reader.dataset,
                                                       split='test'),
            ['first prompt', '\u4f60\u597d\uff0c\u4e16\u754c'])
        template = PromptTemplate(template='{prompt}')
        self.assertEqual(template.generate_item(dataset.test[0]),
                         'first prompt')
        self.assertEqual(ZeroRetriever(dataset).retrieve(), [[], []])


if __name__ == '__main__':
    unittest.main()
