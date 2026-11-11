"""Offline tests for the CivilComments loader and configs."""

import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

from opencompass.datasets import CivilCommentsDataset
from opencompass.openicl.icl_evaluator import AccEvaluator, AUCROCEvaluator
from opencompass.openicl.icl_inferencer import CLPInferencer, GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.utils.text_postprocessors import first_option_postprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PACKAGE = 'opencompass.configs.datasets.civilcomments'

SUBTYPE_COLUMNS = [
    'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack',
    'sexual_explicit'
]


def _rows(*items):
    return Dataset.from_list([
        dict(text=text,
             toxicity=toxicity,
             **dict.fromkeys(SUBTYPE_COLUMNS, 0.0)) for text, toxicity in items
    ])


class TestCivilCommentsDataset(unittest.TestCase):

    def test_load_maps_threshold_to_consistent_numeric_and_ab_labels(self):

        def fake_load_dataset(**kwargs):
            if kwargs['split'] == 'train':
                return _rows(('train', 0.1))
            # The loader subsamples exactly 10000 test rows.
            return _rows(('safe', 0.4999), ('boundary', 0.5), ('toxic', 0.9),
                         *[('filler', 0.0)] * 9997)

        with patch('opencompass.datasets.civilcomments.load_dataset',
                   fake_load_dataset):
            dataset = CivilCommentsDataset.load(path='civil_comments')

        self.assertEqual(len(dataset['test']), 10000)
        self.assertEqual(
            sorted((row['text'], row['label'], row['answer'], row['choices'])
                   for row in dataset['test'] if row['text'] != 'filler'),
            [
                ('boundary', 1, 'B', ['no', 'yes']),
                ('safe', 0, 'A', ['no', 'yes']),
                ('toxic', 1, 'B', ['no', 'yes']),
            ],
        )


class TestCivilCommentsConfigs(unittest.TestCase):

    def test_gen_config_is_strict_ab_accuracy_config(self):
        module = importlib.import_module(f'{CONFIG_PACKAGE}.civilcomments_gen')
        dataset = module.civilcomments_datasets[0]
        prompt_template = dataset['infer_cfg']['prompt_template']
        messages = prompt_template['messages']
        prompt = messages[0]['content']
        postprocessor = dataset['eval_cfg']['pred_postprocessor']

        self.assertEqual(dataset['reader_cfg']['output_column'], 'answer')
        self.assertIs(prompt_template['type'], RawPromptTemplate)
        self.assertEqual([m['role'] for m in messages], ['user'])
        self.assertIs(dataset['infer_cfg']['inferencer']['type'],
                      GenInferencer)
        self.assertIs(dataset['eval_cfg']['evaluator']['type'], AccEvaluator)
        self.assertIn('A. No\nB. Yes', prompt)
        self.assertIn('exactly one letter, A or B', prompt)
        self.assertIs(postprocessor['type'], first_option_postprocess)
        self.assertEqual(postprocessor['options'], 'AB')
        self.assertFalse(postprocessor['cushion'])

    def test_clp_stable_config_remains_auc_based(self):
        module = importlib.import_module(f'{CONFIG_PACKAGE}.civilcomments_clp')
        dataset = module.civilcomments_datasets[0]

        self.assertEqual(dataset['reader_cfg']['output_column'], 'label')
        self.assertIs(dataset['infer_cfg']['inferencer']['type'],
                      CLPInferencer)
        self.assertIs(dataset['eval_cfg']['evaluator']['type'],
                      AUCROCEvaluator)

    def test_dataset_index_exposes_both_stable_configs(self):
        dataset_index = (REPO_ROOT / 'dataset-index.yml').read_text()

        self.assertIn(
            'opencompass/configs/datasets/civilcomments/civilcomments_clp.py',
            dataset_index,
        )
        self.assertIn(
            'opencompass/configs/datasets/civilcomments/civilcomments_gen.py',
            dataset_index,
        )


if __name__ == '__main__':
    unittest.main()
