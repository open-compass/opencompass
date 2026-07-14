"""Unit tests for DefaultSummarizer."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from mmengine.config import ConfigDict

from opencompass.summarizers.default import DefaultSummarizer


class TestDefaultSummarizer(unittest.TestCase):
    """Test cases for DefaultSummarizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConfigDict({
            'work_dir':
            tempfile.mkdtemp(),
            'models': [ConfigDict({'abbr': 'test_model'})],
            'datasets': [ConfigDict({'abbr': 'test_dataset'})]
        })

    def test_initialization(self):
        """Test DefaultSummarizer initialization."""
        summarizer = DefaultSummarizer(config=self.config)
        self.assertEqual(summarizer.cfg, self.config)
        self.assertIsNotNone(summarizer.logger)

    def test_initialization_with_dataset_abbrs(self):
        """Test DefaultSummarizer initialization with dataset_abbrs."""
        dataset_abbrs = ['dataset1', 'dataset2']
        summarizer = DefaultSummarizer(config=self.config,
                                       dataset_abbrs=dataset_abbrs)
        self.assertEqual(summarizer.dataset_abbrs, dataset_abbrs)

    def test_initialization_with_summary_groups(self):
        """Test DefaultSummarizer initialization with summary_groups."""
        summary_groups = [{
            'name': 'test_group',
            'subsets': ['dataset1', 'dataset2']
        }]
        summarizer = DefaultSummarizer(config=self.config,
                                       summary_groups=summary_groups)
        self.assertEqual(summarizer.summary_groups, summary_groups)

    def test_initialization_deprecates_prompt_db(self):
        """Test that prompt_db parameter is deprecated."""
        with patch(
                'opencompass.summarizers.default.get_logger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log

            DefaultSummarizer(config=self.config, prompt_db='deprecated_value')

            # Should log a warning about prompt_db being deprecated
            mock_log.warning.assert_called()


if __name__ == '__main__':
    unittest.main()
