import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from opencompass.datasets.safety import SafetyDataset


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


if __name__ == '__main__':
    unittest.main()
