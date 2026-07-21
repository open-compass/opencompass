import unittest
from unittest.mock import patch

from datasets import Dataset

from opencompass.datasets.boolq import BoolQDataset


class TestBoolQDataset(unittest.TestCase):

    @patch('opencompass.datasets.boolq.load_dataset')
    @patch('opencompass.datasets.boolq.get_data_path')
    def test_resolves_data_files_with_opencompass_mapping(
            self, mock_get_data_path, mock_load_dataset):
        mock_get_data_path.return_value = './data/SuperGLUE/BoolQ/val.jsonl'
        mock_load_dataset.return_value = Dataset.from_list([
            {
                'label': 'true'
            },
            {
                'label': 'false'
            },
        ])

        dataset = BoolQDataset.load(
            path='json', data_files='opencompass/boolq', split='train')

        mock_get_data_path.assert_called_once_with('opencompass/boolq')
        mock_load_dataset.assert_called_once_with(
            path='json',
            data_files='./data/SuperGLUE/BoolQ/val.jsonl',
            split='train')
        self.assertEqual(dataset[0]['answer'], 1)
        self.assertEqual(dataset[1]['answer'], 0)


if __name__ == '__main__':
    unittest.main()
