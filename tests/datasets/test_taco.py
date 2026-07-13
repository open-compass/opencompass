import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from opencompass.datasets.taco import TACODataset


def make_sample(difficulty='EASY', starter_code=''):
    return {
        'question': 'Return the input.',
        'solutions': '[]',
        'starter_code': starter_code,
        'input_output': '{"inputs": ["1\\n"], "outputs": ["1\\n"]}',
        'difficulty': difficulty,
    }


class TestTACODataset(unittest.TestCase):

    @patch('opencompass.datasets.taco.Dataset.from_file')
    @patch('opencompass.datasets.taco.hf_hub_download')
    def test_official_hf_path_loads_test_arrow(self, mock_download,
                                               mock_from_file):
        mock_download.return_value = '/tmp/taco-test.arrow'
        mock_from_file.return_value = Dataset.from_list([make_sample()])

        dataset = TACODataset.load('BAAI/TACO')

        mock_download.assert_called_once_with(
            repo_id='BAAI/TACO',
            repo_type='dataset',
            filename='test/data-00000-of-00001.arrow',
        )
        mock_from_file.assert_called_once_with('/tmp/taco-test.arrow')
        self.assertEqual(len(dataset['test']), 1)
        self.assertEqual(dataset['test'][0]['starter'],
                         '\\nUse Standard Input format')

    @patch('opencompass.datasets.taco.load_from_disk')
    @patch('opencompass.datasets.taco.get_data_path')
    def test_local_path_keeps_local_download_flow(self, mock_get_data_path,
                                                  mock_load_from_disk):
        mock_get_data_path.return_value = '/tmp/BAAI-TACO'
        mock_load_from_disk.return_value = DatasetDict(
            {'test': Dataset.from_list([make_sample(starter_code='def f():')])}
        )

        dataset = TACODataset.load('./data/BAAI-TACO')

        mock_get_data_path.assert_called_once_with('./data/BAAI-TACO',
                                                   local_mode=True)
        mock_load_from_disk.assert_called_once_with('/tmp/BAAI-TACO')
        self.assertEqual(dataset['test'][0]['starter'],
                         'def f():\\nUse Call-Based format')


if __name__ == '__main__':
    unittest.main()
