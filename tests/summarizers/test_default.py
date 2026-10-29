"""Unit tests for DefaultSummarizer."""

import os.path as osp
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import mmengine
from mmengine.config import ConfigDict

from opencompass.summarizers.default import DefaultSummarizer
from opencompass.utils.result import RESULT_METADATA_KEY, SAMPLE_COUNT_KEY


class TestDefaultSummarizer(unittest.TestCase):
    """Test cases for DefaultSummarizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ConfigDict({
            'work_dir':
            self.temp_dir,
            'models': [ConfigDict({'abbr': 'test_model'})],
            'datasets': [
                ConfigDict({
                    'abbr': 'test_dataset',
                    'infer_cfg': {
                        'inferencer': {
                            'type': 'GenInferencer'
                        }
                    }
                })
            ]
        })

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def dump_result(self, dataset_abbr, result):
        """Dump a result file for the default test model."""
        result_dir = osp.join(self.temp_dir, 'results', 'test_model')
        mmengine.mkdir_or_exist(result_dir)
        mmengine.dump(result, osp.join(result_dir, f'{dataset_abbr}.json'))

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

    def test_pick_up_results_combines_split_results_with_details(self):
        """Test summarizer combines legacy split result files using details."""
        self.dump_result('test_dataset_0', {
            'accuracy': 80.0,
            'details': [{}, {}],
        })
        self.dump_result('test_dataset_1', {
            'accuracy': 100.0,
            'details': [{}],
        })

        summarizer = DefaultSummarizer(config=self.config)
        raw_results, parsed_results, dataset_metrics, _ = \
            summarizer._pick_up_results()

        self.assertAlmostEqual(
            parsed_results['test_model']['test_dataset']['accuracy'],
            86.6666666667)
        self.assertEqual(dataset_metrics['test_dataset'], ['accuracy'])
        self.assertNotIn('details',
                         raw_results['test_model']['test_dataset'])

    def test_pick_up_results_combines_split_results_with_metadata(self):
        """Test summarizer combines split result files without details."""
        self.dump_result('test_dataset_0', {
            'accuracy': 50.0,
            RESULT_METADATA_KEY: {
                SAMPLE_COUNT_KEY: 2
            },
        })
        self.dump_result('test_dataset_1', {
            'accuracy': 100.0,
            RESULT_METADATA_KEY: {
                SAMPLE_COUNT_KEY: 1
            },
        })

        summarizer = DefaultSummarizer(config=self.config)
        raw_results, parsed_results, _, _ = summarizer._pick_up_results()

        self.assertAlmostEqual(
            parsed_results['test_model']['test_dataset']['accuracy'],
            66.6666666667)
        self.assertNotIn(RESULT_METADATA_KEY,
                         raw_results['test_model']['test_dataset'])

    def test_pick_up_results_prefers_complete_result_file(self):
        """Test unsplit result files take precedence over split files."""
        self.dump_result('test_dataset', {'accuracy': 70.0})
        self.dump_result('test_dataset_0', {
            'accuracy': 50.0,
            RESULT_METADATA_KEY: {
                SAMPLE_COUNT_KEY: 1
            },
        })

        summarizer = DefaultSummarizer(config=self.config)
        _, parsed_results, _, _ = summarizer._pick_up_results()

        self.assertEqual(
            parsed_results['test_model']['test_dataset']['accuracy'], 70.0)


if __name__ == '__main__':
    unittest.main()
