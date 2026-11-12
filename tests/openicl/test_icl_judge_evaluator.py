# flake8: noqa
from opencompass.openicl.icl_evaluator.icl_judge_evaluator import JudgeEvaluator


def test_judge_evaluator_truncated_choice_does_not_crash():
    """A judge output truncated right at the ``"Choice": "Model `` marker (before
    the model letter) must not raise IndexError; the missing choice should simply
    not match the gold winner."""
    evaluator = JudgeEvaluator()
    predictions = ['{"Choice": "Model ']  # truncated before the letter
    references = [{'winner': 'A'}]

    result = evaluator.score(predictions, references)

    assert result['accuracy'] == 0.0
    assert result['details'][0]['correct'] is False


def test_judge_evaluator_scores_normal_choice():
    evaluator = JudgeEvaluator()
    predictions = ['Reasoning... {"Choice": "Model A"}']
    references = [{'winner': 'A'}]

    result = evaluator.score(predictions, references)

    assert result['accuracy'] == 100.0
    assert result['details'][0]['correct'] is True
