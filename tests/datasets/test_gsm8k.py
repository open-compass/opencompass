import copy
import os
import unittest
from unittest.mock import patch

from opencompass.configs.datasets.gsm8k.gsm8k_cascade_eval_rawprompt_gen_36fce7 import \
    gsm8k_datasets
from opencompass.datasets.gsm8k import (GSM8KDataset,
                                        gsm8k_dataset_postprocess,
                                        gsm8k_postprocess)
from opencompass.evaluator import (CascadeEvaluator, GenericLLMEvaluator,
                                   MATHVerifyEvaluator)
from opencompass.registry import ICL_EVALUATORS


class TestGSM8KCascadeConfig(unittest.TestCase):

    def test_cascade_config_uses_rule_postprocess_only(self):
        dataset_cfg = gsm8k_datasets[0]
        eval_cfg = dataset_cfg['eval_cfg']
        evaluator_cfg = eval_cfg['evaluator']

        self.assertIs(dataset_cfg['type'], GSM8KDataset)
        self.assertIs(eval_cfg['dataset_postprocessor']['type'],
                      gsm8k_dataset_postprocess)
        self.assertNotIn('pred_postprocessor', eval_cfg)
        self.assertIs(evaluator_cfg['type'], CascadeEvaluator)
        self.assertFalse(evaluator_cfg['parallel'])
        self.assertIs(evaluator_cfg['rule_evaluator']['type'],
                      MATHVerifyEvaluator)
        self.assertIs(
            evaluator_cfg['rule_evaluator']['pred_postprocessor']['type'],
            gsm8k_postprocess)
        self.assertIs(evaluator_cfg['llm_evaluator']['type'],
                      GenericLLMEvaluator)

    def test_cascade_evaluator_builds(self):
        evaluator_cfg = copy.deepcopy(
            gsm8k_datasets[0]['eval_cfg']['evaluator'])

        judge_env = {
            'OC_JUDGE_MODEL': 'dummy-judge',
            'OC_JUDGE_API_KEY': 'dummy-key',
            'OC_JUDGE_API_BASE': 'http://localhost/v1',
        }
        with patch.dict(os.environ, judge_env):
            evaluator = ICL_EVALUATORS.build(evaluator_cfg)

        self.assertIsInstance(evaluator, CascadeEvaluator)
        self.assertIsInstance(evaluator.rule_evaluator, MATHVerifyEvaluator)
        self.assertIsInstance(evaluator.llm_evaluator, GenericLLMEvaluator)
        self.assertFalse(evaluator.parallel)


if __name__ == '__main__':
    unittest.main()
