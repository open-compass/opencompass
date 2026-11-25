"""Tests for repeat-analysis summaries, including empty prediction runs."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mmengine import ConfigDict

from opencompass.utils.repeat_analysis import (
    _collect_predictions_by_benchmark, _format_analysis_summary_table,
    print_analysis_summary)


class TestRepeatAnalysisSummary(unittest.TestCase):

    @staticmethod
    def _report(stats):
        return {
            'benchmark_stats': stats,
            'abnormal_samples': {
                'repeat_pattern': [],
                'gzip_high_compression': [],
            },
        }

    def test_empty_summary_has_zero_overall_row(self):
        table = _format_analysis_summary_table(self._report({}))
        self.assertEqual(len(table), 2)
        self.assertEqual(table[-1],
                         ['Overall', '0', '0', '0.00% (0)', '0.00% (0)'])

    def test_print_empty_summary(self):
        output = io.StringIO()
        with redirect_stdout(output):
            print_analysis_summary(self._report({}))
        self.assertIn('Repeat analysis summary:', output.getvalue())
        self.assertIn('Overall', output.getvalue())
        self.assertNotIn('reasoning length', output.getvalue())

    def test_missing_prediction_files(self):
        with tempfile.TemporaryDirectory() as work_dir:
            config = ConfigDict(models=[dict(abbr='model')],
                                datasets=[dict(abbr='missing')],
                                work_dir=work_dir)
            records, missing = _collect_predictions_by_benchmark(config)
            self.assertEqual(dict(records), {})
            self.assertEqual(len(missing), 1)
            self.assertEqual(missing[0]['benchmark'], 'missing')
            table = _format_analysis_summary_table(self._report(records))
            self.assertEqual(table[-1][1], '0')

    def test_empty_prediction_file(self):
        with tempfile.TemporaryDirectory() as work_dir:
            path = Path(work_dir) / 'predictions' / 'model' / 'empty.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({}), encoding='utf-8')
            config = ConfigDict(models=[dict(abbr='model')],
                                datasets=[dict(abbr='empty')],
                                work_dir=work_dir)
            records, missing = _collect_predictions_by_benchmark(config)
            self.assertEqual(dict(records), {})
            self.assertEqual(missing, [])
            table = _format_analysis_summary_table(self._report(records))
            self.assertEqual(table[-1][1], '0')

    def test_nonempty_summary_preserves_columns_and_totals(self):
        stats = {
            'sample_count': 2,
            'length': {
                'mean': 10,
                'p75': 12,
                'p90': 14
            },
        }
        table = _format_analysis_summary_table(self._report({'sample': stats}))
        self.assertEqual(
            table[1],
            ['sample', '2', '10 / 12 / 14', '0.00% (0)', '0.00% (0)'])
        self.assertEqual(table[-1],
                         ['Overall', '2', '10', '0.00% (0)', '0.00% (0)'])

    def test_reasoning_summary_preserves_columns_and_totals(self):
        stats = {
            'sample_count': 2,
            'length': {
                'mean': 10,
                'p75': 12,
                'p90': 14
            },
            'reasoning_length': {
                'mean': 7
            },
            'content_length': {
                'mean': 3
            },
        }
        table = _format_analysis_summary_table(self._report({'sample': stats}))
        self.assertIn('reasoning length', table[0])
        self.assertEqual(table[1][3:5], [7, 3])
        self.assertEqual(table[-1][1:5], ['2', '10', '7', '3'])
