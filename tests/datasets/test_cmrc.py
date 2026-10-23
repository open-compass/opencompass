"""CMRC exact-match regressions through the real dataset and configurations."""

import json
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset
from mmengine.config import Config

from opencompass.datasets import CMRCDataset
from opencompass.registry import ICL_EVALUATORS

CONFIG_DIR = (Path(__file__).resolve().parents[2] / 'opencompass' / 'configs' /
              'datasets' / 'CLUE_CMRC')
CONFIGS = [
    'CLUE_CMRC_gen_1bd3c8.py',
    'CLUE_CMRC_gen_3749cd.py',
    'CLUE_CMRC_gen_8484b9.py',
    'CLUE_CMRC_gen_941108.py',
]

# Literal expectations distinguish full answer spans. The official CMRC EM
# protocol lowercases and strips outer whitespace, removes its selected
# punctuation, and preserves ASCII periods, commas and internal whitespace.
CASES = [
    ('北京', ['北京'], True),
    ('上海', ['北京'], False),
    ('3.14', ['3.14'], True),
    ('3.15', ['3.14'], False),
    ('3', ['3.14'], False),
    ('St. Louis', ['St. Louis'], True),
    ('St. Paul', ['St. Louis'], False),
    ('Portland, Maine', ['Portland, Oregon'], False),
    ('中国', ['中华人民共和国', '中国'], True),
    ('《红楼梦》', ['红楼梦'], True),
    (' PARIS ', ['Paris'], True),
    ('man', ['a man'], False),
    ('甲  乙', ['甲 乙'], False),
    ('甲乙', ['甲。乙'], True),
    ('甲乙', ['甲……乙'], False),
    ('abcd', ['ab\ncd'], False),
]


class TestCMRC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / 'cmrc.json'
        path.write_text(json.dumps({
            'data': [{
                'paragraphs': [{
                    'context':
                    'Synthetic context for local scoring.',
                    'qas': [{
                        'id':
                        str(i),
                        'question':
                        f'Question {i}',
                        'answers': [{
                            'text': answer
                        } for answer in answers],
                    } for i, (_, answers, _) in enumerate(CASES)],
                }],
            }],
        }),
                        encoding='utf-8')
        cls.dataset = CMRCDataset(path=str(path),
                                  abbr='cmrc_fixture',
                                  reader_cfg=dict(
                                      input_columns=['question', 'context'],
                                      output_column='answers'))
        cls.configs = {
            name: Config.fromfile(str(CONFIG_DIR / name))
            for name in CONFIGS
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def evaluator(self, name):
        evaluator_type = self.configs[name].CMRC_datasets[0]['eval_cfg'][
            'evaluator']['type']
        # Build by the configured class name to exercise its registration too.
        return ICL_EVALUATORS.build(dict(type=evaluator_type.__name__))

    def test_loader_retains_full_aliases(self):
        self.assertIsInstance(self.dataset.dataset, Dataset)
        self.assertEqual(len(self.dataset.dataset), len(CASES))
        for row, (_, answers, _) in zip(self.dataset.dataset, CASES):
            self.assertEqual(set(row['answers']), set(answers))
            self.assertEqual(row['context'],
                             'Synthetic context for local scoring.')

    def test_all_configs_score_full_answers(self):
        for name in CONFIGS:
            evaluator = self.evaluator(name)
            for i, (prediction, _, correct) in enumerate(CASES):
                with self.subTest(config=name, prediction=prediction):
                    answers = self.dataset.dataset[i]['answers']
                    result = evaluator.score([prediction], [answers])
                    self.assertEqual(result['score'], 100 * int(correct))
                    self.assertEqual(result['details'][0]['correct'], correct)

    def test_public_evaluate_keeps_every_row(self):
        predictions = [prediction for prediction, _, _ in CASES]
        references = self.dataset.dataset['answers']
        expected = 100 * sum(correct for _, _, correct in CASES) / len(CASES)
        for name in CONFIGS:
            with self.subTest(config=name):
                result = self.evaluator(name).evaluate(
                    k=1,
                    n=1,
                    original_dataset=self.dataset.dataset,
                    predictions=predictions,
                    references=references)
                self.assertEqual(result['score'], expected)
                self.assertEqual(len(result['details']), len(CASES))

    def test_details_preserve_complete_decimal(self):
        for name in CONFIGS:
            with self.subTest(config=name):
                result = self.evaluator(name).score(['3.14'], [['3.14']])
                self.assertEqual(result['details'][0]['pred'], '3.14')
                self.assertIn('3.14', result['details'][0]['answer'])

    def test_missing_alias_and_empty_reply(self):
        for name in CONFIGS:
            with self.subTest(config=name):
                result = self.evaluator(name).score(['', 'x', ''],
                                                    [['北京'], [], ['']])
                self.assertAlmostEqual(result['score'], 100 / 3)
                self.assertEqual([d['correct'] for d in result['details']],
                                 [False, False, True])

    def test_length_mismatch_returns_error(self):
        for name in CONFIGS:
            with self.subTest(config=name):
                self.assertIn('error', self.evaluator(name).score(['北京'], []))

    def test_empty_dataset_reports_error(self):
        for name in CONFIGS:
            with self.subTest(config=name):
                self.assertIn('error', self.evaluator(name).score([], []))

    def test_default_config_alias_scores_cmrc(self):
        config = Config.fromfile(str(CONFIG_DIR / 'CLUE_CMRC_gen.py'))
        evaluator_type = config.CMRC_datasets[0]['eval_cfg']['evaluator'][
            'type']
        result = ICL_EVALUATORS.build(dict(type=evaluator_type)).score(
            ['3.15'], [['3.14']])
        self.assertEqual(result['score'], 0)


if __name__ == '__main__':
    unittest.main()
