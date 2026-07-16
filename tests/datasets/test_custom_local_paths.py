import unittest
from unittest.mock import patch

from opencompass.utils.datasets import get_data_path


class TestCustomDatasetLocalPaths(unittest.TestCase):

    def test_platform_absolute_path_is_returned_directly(self):
        path = 'C:\\data\\local_dataset.csv'
        with patch('opencompass.utils.datasets.os.path.isabs',
                   return_value=True) as isabs:
            self.assertEqual(get_data_path(path), path)

        isabs.assert_called_once_with(path)


if __name__ == '__main__':
    unittest.main()
