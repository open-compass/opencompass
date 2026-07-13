import os
import tempfile
import unittest
from unittest.mock import patch

from opencompass.datasets.custom import CustomDataset


class TestCustomDatasetLocalPaths(unittest.TestCase):

    def test_existing_relative_path_takes_precedence_over_cache(self):
        with tempfile.TemporaryDirectory() as work_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                data_path = os.path.join(work_dir, 'local_dataset.csv')
                with open(data_path, 'w', encoding='utf-8') as f:
                    f.write('question,A,B,C,answer\n1+1=?,1,2,3,B\n')

                original_cwd = os.getcwd()
                try:
                    os.chdir(work_dir)
                    with patch.dict(os.environ,
                                    {'COMPASS_DATA_CACHE': cache_dir},
                                    clear=True):
                        dataset = CustomDataset.load('local_dataset.csv')
                finally:
                    os.chdir(original_cwd)

                self.assertEqual(len(dataset), 1)
                self.assertEqual(dataset[0]['answer'], 'B')


if __name__ == '__main__':
    unittest.main()
