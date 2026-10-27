from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator


class _AccuracyEvaluator(BaseEvaluator):
    """Returns only an aggregate score, like AccEvaluator."""

    def score(self, predictions, references):
        correct = sum(p == r for p, r in zip(predictions, references))
        return {'accuracy': 100 * correct / len(references)}


class _AccuracyWithDetailsEvaluator(_AccuracyEvaluator):

    def score(self, predictions, references):
        result = super().score(predictions, references)
        result['details'] = [{
            'pred': p,
            'answer': r,
            'correct': p == r
        } for p, r in zip(predictions, references)]
        return result


def _two_runs(evaluator):
    # 4 examples repeated n=2 times: the first run gets all 4 right,
    # the second run gets 1 of 4
    dataset = Dataset.from_dict({
        'subdivision': ['demo'] * 8,
        'idx': [0, 1, 2, 3] * 2,
    })
    result = evaluator.evaluate(
        k=1,
        n=2,
        original_dataset=dataset,
        predictions=['a', 'b', 'c', 'd', 'a', 'x', 'x', 'x'],
        references=['a', 'b', 'c', 'd'] * 2,
    )
    result.pop('details', None)
    return result


def test_n_runs_are_averaged_without_details():
    assert _two_runs(_AccuracyEvaluator()) == {
        'accuracy (2 runs average)': 62.5
    }


def test_n_runs_are_averaged_with_details():
    assert _two_runs(_AccuracyWithDetailsEvaluator()) == {
        'accuracy (2 runs average)': 62.5
    }


def test_n_runs_error_is_returned_as_is():

    class _FailingEvaluator(BaseEvaluator):

        def score(self, predictions, references):
            return {'error': 'service unavailable'}

    assert _two_runs(_FailingEvaluator()) == {'error': 'service unavailable'}


def test_single_run_result_is_returned_as_is():
    dataset = Dataset.from_dict({'subdivision': ['demo'], 'idx': [0]})
    result = _AccuracyEvaluator().evaluate(
        k=1,
        n=1,
        original_dataset=dataset,
        predictions=['a'],
        references=['a'],
    )
    assert result == {'accuracy': 100.0}
