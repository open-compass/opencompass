import unittest
from unittest.mock import patch

from datasets import Dataset

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

    def test_local_path_is_not_supported(self):
        with self.assertRaisesRegex(AssertionError,
                                    'only supports the Hugging Face dataset'):
            TACODataset.load('./data/BAAI-TACO')


if __name__ == '__main__':
    unittest.main()
