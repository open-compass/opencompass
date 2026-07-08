import json
import tempfile
import unittest
from unittest.mock import patch

from opencompass.datasets.mbpp import MBPPPlusDataset


class TestMBPPPlusDataset(unittest.TestCase):

    def test_loads_evalplus_schema(self):
        sample = {
            'task_id': 'Mbpp/2',
            'prompt': '"""\nWrite a function to find shared elements.\n'
            'assert set(foo([1], [1])) == set([1])\n"""',
            'assertion': '\nassert foo([1], [1]) == [1]\n'
            'assert foo([1], [2]) == []\n',
            'base_input': [],
            'plus_input': [],
            'canonical_solution': '',
            'contract': '',
            'atol': 0,
        }

        with tempfile.NamedTemporaryFile('w',
                                         encoding='utf-8',
                                         suffix='.jsonl') as f:
            f.write(json.dumps(sample) + '\n')
            f.flush()

            with patch('opencompass.datasets.mbpp.get_data_path',
                       return_value=f.name):
                dataset = MBPPPlusDataset.load('opencompass/mbpp_plus')

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]['task_id'], 'Mbpp/2')
        self.assertEqual(dataset[0]['text'],
                         'Write a function to find shared elements.')
        self.assertEqual(dataset[0]['test_case'], [
            'assert foo([1], [1]) == [1]',
            'assert foo([1], [2]) == []',
        ])
        self.assertEqual(
            dataset[0]['test_list'],
            'assert foo([1], [1]) == [1]\nassert foo([1], [2]) == []')


if __name__ == '__main__':
    unittest.main()
