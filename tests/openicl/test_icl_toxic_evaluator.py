import itertools

import pytest

from opencompass.openicl.icl_evaluator.icl_toxic_evaluator import \
    ToxicEvaluator


def _evaluator():
    """Build a ToxicEvaluator without constructing the API client."""
    evaluator = ToxicEvaluator.__new__(ToxicEvaluator)
    evaluator.thr = 0.5
    return evaluator


@pytest.mark.parametrize(
    "scores",
    [scores for scores in itertools.permutations((0.1, 0.9, -1.0))],
)
def test_expected_max_toxicity_ignores_invalid_scores_and_order(scores):
    evaluator = _evaluator()
    response = {
        str(index): {
            "toxicity_score": score
        }
        for index, score in enumerate(scores)
    }

    metrics = evaluator.get_metrics(response)

    assert metrics["expected_max_toxicity"] == 0.9
    assert metrics["valid_frac"] == round(2 / 3, 4)
    assert metrics["toxic_frac_valid"] == 0.5
    assert metrics["avg_toxicity_score"] == 0.5


def test_expected_max_toxicity_still_rejects_all_invalid_scores():
    evaluator = _evaluator()
    response = {
        "first": {
            "toxicity_score": -1.0
        },
        "second": {
            "toxicity_score": -1.0
        },
    }

    with pytest.raises(ValueError, match="All predictions are not valid"):
        evaluator.get_metrics(response)
