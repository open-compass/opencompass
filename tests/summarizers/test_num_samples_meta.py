"""Unit tests for comparability metadata (num_samples / test_range).

Covers: the eval task attaching `num_samples` / `test_range` to results,
and the default summarizer annotating summary-table rows whose models
were evaluated on different sample subsets.
"""

import unittest

from mmengine.config import ConfigDict

from opencompass.summarizers.default import DefaultSummarizer
from opencompass.tasks.openicl_eval import OpenICLEvalTask


class TestAttachDatasetMeta(unittest.TestCase):

    def _task(self, dataset_cfg):
        task = OpenICLEvalTask.__new__(OpenICLEvalTask)
        task.dataset_cfg = dataset_cfg
        return task

    def test_num_samples_divides_by_n(self):
        """The test set is n-replicated; num_samples must report the
        single-replica size."""
        task = self._task(ConfigDict({'n': 3}))
        result = {'accuracy': 50.0}

        task._attach_dataset_meta(result, ['sample'] * 9)

        self.assertEqual(result['num_samples'], 3)
        self.assertNotIn('test_range', result)

    def test_test_range_passthrough(self):
        task = self._task(
            ConfigDict({'n': 1,
                        'reader_cfg': {'test_range': '0:10'}}))
        result = {}

        task._attach_dataset_meta(result, ['a'] * 10)

        self.assertEqual(result['num_samples'], 10)
        self.assertEqual(result['test_range'], '0:10')

    def test_no_overwrite_and_non_dict_result(self):
        task = self._task(ConfigDict({'n': 1}))
        result = {'num_samples': 42}

        task._attach_dataset_meta(result, ['a', 'b'])
        task._attach_dataset_meta('not-a-dict', ['a'])

        self.assertEqual(result['num_samples'], 42)


def _make_summarizer():
    summarizer = DefaultSummarizer.__new__(DefaultSummarizer)
    summarizer.model_abbrs = ['m1', 'm2']
    summarizer.dataset_cfgs = [
        ConfigDict({
            'abbr': 'demo',
            'infer_cfg': {'retriever': {'type': 'FixKRetriever'}}
        })
    ]
    return summarizer


def _table_inputs():
    parsed_results = {
        'm1': {'demo': {'accuracy': 60.0}},
        'm2': {'demo': {'accuracy': 80.0}},
    }
    dataset_metrics = {'demo': ['accuracy']}
    dataset_eval_mode = {'demo': 'gen'}
    return parsed_results, dataset_metrics, dataset_eval_mode


class TestTableAnnotation(unittest.TestCase):

    def test_comparable_meta_not_annotated(self):
        summarizer = _make_summarizer()
        parsed_results, dataset_metrics, dataset_eval_mode = _table_inputs()
        meta = {
            'm1': {'demo': {'num_samples': 100}},
            'm2': {'demo': {'num_samples': 100}},
        }

        table = summarizer._format_table(parsed_results, dataset_metrics,
                                         dataset_eval_mode,
                                         dataset_meta=meta)

        self.assertEqual(table[1][0], 'demo')
        self.assertEqual(len(table), 2)

    def test_different_num_samples_annotated(self):
        summarizer = _make_summarizer()
        parsed_results, dataset_metrics, dataset_eval_mode = _table_inputs()
        meta = {
            'm1': {'demo': {'num_samples': 100}},
            'm2': {'demo': {'num_samples': 5}},
        }

        table = summarizer._format_table(parsed_results, dataset_metrics,
                                         dataset_eval_mode,
                                         dataset_meta=meta)

        self.assertEqual(table[1][0], 'demo*')
        self.assertEqual(len(table), 3)
        self.assertIn('not directly comparable', table[2][0])

    def test_different_test_range_annotated(self):
        summarizer = _make_summarizer()
        parsed_results, dataset_metrics, dataset_eval_mode = _table_inputs()
        meta = {
            'm1': {'demo': {'num_samples': 5, 'test_range': '0:5'}},
            'm2': {'demo': {'num_samples': 5, 'test_range': '5:10'}},
        }

        table = summarizer._format_table(parsed_results, dataset_metrics,
                                         dataset_eval_mode,
                                         dataset_meta=meta)

        self.assertEqual(table[1][0], 'demo*')

    def test_missing_meta_not_annotated(self):
        """Models without metadata must not trigger the annotation."""
        summarizer = _make_summarizer()
        parsed_results, dataset_metrics, dataset_eval_mode = _table_inputs()

        table = summarizer._format_table(parsed_results, dataset_metrics,
                                         dataset_eval_mode,
                                         dataset_meta={'m1': {}})

        self.assertEqual(table[1][0], 'demo')

    def test_single_model_not_annotated(self):
        """One model alone has nothing to be incomparable against."""
        summarizer = _make_summarizer()
        summarizer.model_abbrs = ['m1']
        parsed_results = {'m1': {'demo': {'accuracy': 60.0}}}
        dataset_metrics = {'demo': ['accuracy']}
        dataset_eval_mode = {'demo': 'gen'}
        meta = {'m1': {'demo': {'num_samples': 5}}}

        table = summarizer._format_table(parsed_results, dataset_metrics,
                                         dataset_eval_mode,
                                         dataset_meta=meta)

        self.assertEqual(table[1][0], 'demo')


if __name__ == '__main__':
    unittest.main()
