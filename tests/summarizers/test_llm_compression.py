"""Regression tests for compression summary output."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from mmengine import ConfigDict

from opencompass.summarizers.llm_compression import LLMCompressionSummarizer
from opencompass.utils import get_infer_output_path


class TestLLMCompressionSummarizer(unittest.TestCase):

    def test_pivot_preserves_input_table(self):
        summarizer = LLMCompressionSummarizer(
            ConfigDict(models=[], datasets=[], work_dir='.'))
        table = [
            ['dataset', 'version', 'metric', 'mode', 'model'],
            ['llm_compression-commoncraw', 'af04af', 'bpc', 'unknown', '0.6'],
            ['llm_compression-python', 'af04af', 'bpc', 'unknown', '0.3'],
            ['llm_compression-arxiv_math', 'af04af', 'bpc', 'unknown', '0.45'],
        ]
        original = copy.deepcopy(table)
        first = summarizer._format_table_pivot(table)
        self.assertEqual(table, original)
        second = summarizer._format_table_pivot(table)
        pd.testing.assert_frame_equal(first, second)
        self.assertAlmostEqual(first.iloc[0]['average'], 0.45)

    def test_summarize_preserves_headers_and_all_datasets(self):
        with tempfile.TemporaryDirectory() as work_dir:
            datasets = [
                dict(abbr=f'llm_compression-{name}',
                     infer_cfg=dict(
                         prompt_template=dict(template='{content}'),
                         retriever=dict(type='ZeroRetriever'),
                         inferencer=dict(type='SWCELossInferencer')))
                for name in ('arxiv_math', 'commoncraw', 'python')
            ]
            config = ConfigDict(
                models=[dict(abbr='model'),
                        dict(abbr='better_model')],
                datasets=datasets,
                work_dir=work_dir)
            for model, values in zip(config.models,
                                     ((0.45, 0.6, 0.3), (0.3, 0.4, 0.2))):
                for dataset, value in zip(config.datasets, values):
                    path = Path(
                        get_infer_output_path(model, dataset,
                                              str(Path(work_dir) / 'results')))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({'bpc': value}),
                                    encoding='utf-8')

            LLMCompressionSummarizer(config).summarize(time_str='regression')

            summary_dir = Path(work_dir) / 'summary'
            with (summary_dir / 'summary_regression.csv').open() as file:
                rows = list(csv.reader(file))
            self.assertEqual(rows[0], [
                'dataset', 'version', 'metric', 'mode', 'model', 'better_model'
            ])
            self.assertEqual([row[0] for row in rows[1:]],
                             [dataset.abbr for dataset in config.datasets])
            markdown = (summary_dir / 'summary_regression.md').read_text()
            self.assertIn('dataset', markdown.splitlines()[0])
            self.assertIn('model', markdown.splitlines()[0])
            self.assertIn('llm_compression-arxiv_math',
                          markdown.splitlines()[2])
            text = (summary_dir / 'summary_regression.txt').read_text()
            self.assertIn('dataset,version,metric,mode,model', text)
            pivot = pd.read_csv(summary_dir / 'summary_pivot_regression.csv')
            self.assertEqual(list(pivot['model']), ['better_model', 'model'])
            self.assertAlmostEqual(pivot.iloc[0]['average'], 0.3)
            self.assertAlmostEqual(pivot.iloc[1]['average'], 0.45)
            self.assertEqual(rows[1][-2:], ['0.4500', '0.3000'])
            tabulate_section = text.split('tabulate format\n')[1].split(
                'csv format')[0]
            self.assertIn('dataset', tabulate_section.splitlines()[1])
            self.assertIn('llm_compression-arxiv_math', tabulate_section)


if __name__ == '__main__':
    unittest.main()
