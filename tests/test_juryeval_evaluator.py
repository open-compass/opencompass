"""Tests for JuryEvalEvaluator."""

from unittest.mock import patch

import pytest
from opencompass.evaluator.juryeval_evaluator import JuryEvalEvaluator


class TestJuryEvalEvaluator:
    def test_pairwise_scoring(self):
        evaluator = JuryEvalEvaluator(
            judge_type='pairwise', judge_model='test-model', max_retries=1
        )

        with patch.object(
            evaluator, 'score'
        ) as mock_score:
            mock_score.return_value = {'score': 0.75}
            result = evaluator.score(
                predictions=['ans_a', 'ans_b'],
                references=['ref_a', 'ref_b'],
            )
            assert result['score'] == 0.75

    def test_pointwise_scoring(self):
        evaluator = JuryEvalEvaluator(
            judge_type='pointwise', judge_model='test-model', max_retries=1
        )

        with patch.object(
            evaluator, 'score'
        ) as mock_score:
            mock_score.return_value = {'score': 0.6}
            result = evaluator.score(
                predictions=['ans_a', 'ans_b'],
                references=['ref_a', 'ref_b'],
            )
            assert result['score'] == 0.6

    def test_registered_in_registry(self):
        from opencompass.registry import ICL_EVALUATORS
        assert 'JuryEvalEvaluator' in ICL_EVALUATORS

    def test_unknown_judge_type(self):
        evaluator = JuryEvalEvaluator(
            judge_type='unknown', max_retries=1
        )
        with pytest.raises(ValueError, match='Unknown judge_type'):
            evaluator.score(['a'], ['b'])
