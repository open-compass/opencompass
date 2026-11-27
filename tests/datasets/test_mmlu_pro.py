import pytest

from opencompass.datasets.mmlu_pro import MMLUProBaseEvaluator


@pytest.mark.parametrize('prediction, expected_accuracy', [
    ('B', 100.0),
    ('First sentence. Second sentence.', 100.0),
    ('A', 0.0),
])
def test_score_with_period_in_option_text(prediction, expected_accuracy):
    reference = 'B. First sentence. Second sentence.\n'
    result = MMLUProBaseEvaluator().score([prediction], [reference])
    assert result['accuracy'] == expected_accuracy
