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

    def test_summary_version_is_prompt_hash_prefix(self):
        """Test the version column keeps reporting the prompt hash prefix."""
        dataset = ConfigDict({
            'abbr': 'race-middle_5831a0',
            'version': 'data-v2',
            'infer_cfg': {
                'retriever': {
                    'type': 'ZeroRetriever'
                },
                'inferencer': {
                    'type': 'GenInferencer'
                },
                'prompt_template': {
                    'type': 'PromptTemplate',
                    'template': 'Question: {question}'
                }
            }
        })
        self.config.datasets = [dataset]
        summarizer = DefaultSummarizer(config=self.config)

        table = summarizer._format_table(
            parsed_results={
                'test_model': {
                    'race-middle_5831a0': {
                        'accuracy': 88.5
                    }
                }
            },
            dataset_metrics={'race-middle_5831a0': ['accuracy']},
            dataset_eval_mode={'race-middle_5831a0': 'gen'})

        self.assertEqual(
            table,
            [['dataset', 'version', 'metric', 'mode', 'test_model'],
             ['race-middle_5831a0', '6b938f', 'accuracy', 'gen', '88.50']])
        self.assertNotEqual(table[1][1], '5831a0')
        self.assertNotEqual(table[1][1], dataset.version)


if __name__ == '__main__':
    unittest.main()
