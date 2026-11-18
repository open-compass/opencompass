"""Unit tests for DefaultSummarizer."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import mmengine
from mmengine.config import ConfigDict

from opencompass.summarizers import DefaultSubjectiveSummarizer, needlebench
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

    def _write_summary_results(self, work_dir, result_model):
        datasets = [
            dict(abbr=name,
                 infer_cfg=dict(inferencer=dict(type='GenInferencer'),
                                retriever=dict(type='ZeroRetriever')))
            for name in ('a', 'b', 'c')
        ]
        judge = dict(abbr='judge')
        config = ConfigDict(models=[dict(abbr='test_model')],
                            datasets=datasets,
                            work_dir=work_dir,
                            eval=dict(partitioner=dict(models=[judge])),
                            judge_models=[judge])
        result_dir = Path(work_dir) / 'results' / result_model
        result_dir.mkdir(parents=True)
        sample_results = [('a', 90, 10), ('b', 70, 30), ('c', 60, 40)]
        for name, accuracy, rouge in sample_results:
            mmengine.dump(dict(rouge1=rouge, accuracy=accuracy),
                          str(result_dir / f'{name}.json'))
        return config

    def test_summary_groups_preserve_metric_priority(self):
        """Keep metric priorities for displayed groups and nested averages."""
        cases = (
            (DefaultSummarizer, 'test_model', 'summary.csv'),
            (DefaultSubjectiveSummarizer, 'test_model_judged-by--judge',
             'summary_by_judge.csv'),
        )
        for summarizer_type, result_model, csv_filename in cases:
            with self.subTest(summarizer=summarizer_type.__name__):
                with tempfile.TemporaryDirectory() as work_dir:
                    config = self._write_summary_results(
                        work_dir, result_model)

                    summarizer = summarizer_type(
                        config,
                        dataset_abbrs=[
                            'a', 'child', ('child', 'rouge1'),
                            ('parent', 'naive_average')
                        ],
                        summary_groups=[
                            dict(name='child', subsets=['a', 'b']),
                            dict(name='parent',
                                 subsets=['child', 'c'],
                                 metric='naive_average')
                        ])
                    summarizer.summarize(
                        output_path=str(Path(work_dir) / 'summary.txt'))

                    with (Path(work_dir) /
                          csv_filename).open(newline='') as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(
                        [(row['dataset'], row['metric'], row['test_model'])
                         for row in rows], [
                             ('a', 'accuracy', '90.00'),
                             ('child', 'accuracy', '80.00'),
                             ('child', 'rouge1', '20.00'),
                             ('parent', 'naive_average', '70.00'),
                         ])

    def test_needlebench_tables_preserve_metric_priority(self):
        """NeedleBench tables retain the result reader's metric priorities."""
        for summarizer_type in (needlebench.NeedleBenchSummarizer,
                                needlebench.NeedleBenchATCSummarizer):
            with self.subTest(summarizer=summarizer_type.__name__):
                with tempfile.TemporaryDirectory() as work_dir:
                    config = self._write_summary_results(
                        work_dir, 'test_model')
                    summarizer = summarizer_type(
                        config,
                        dataset_abbrs=[
                            'a', 'child', ('child', 'rouge1'),
                            ('parent', 'naive_average')
                        ],
                        summary_groups=[
                            dict(name='child', subsets=['a', 'b']),
                            dict(name='parent',
                                 subsets=['child', 'c'],
                                 metric='naive_average')
                        ])
                    results = summarizer._pick_up_results()
                    _, parsed, metrics, modes = \
                        summarizer._calculate_group_metrics(*results)
                    table = summarizer._format_table(parsed, metrics, modes)
                    self.assertEqual(table, [
                        ['dataset', 'test_model'],
                        ['a', '90.00'],
                        ['child', '80.00'],
                        ['child', '20.00'],
                        ['parent', '70.00'],
                    ])

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
