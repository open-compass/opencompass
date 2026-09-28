from datasets import Dataset

from opencompass.openicl.icl_evaluator import AveragePPLEvaluator


def test_average_ppl_evaluator_without_predictions():
    dataset = Dataset.from_list([{'text': 'hello'}])

    result = AveragePPLEvaluator().evaluate(
        k=1,
        n=1,
        original_dataset=dataset,
        ppl=[2.0],
    )

    assert result == {'average_ppl': 2.0}


def test_average_ppl_evaluator_aggregates_multiple_replicas():
    dataset = Dataset.from_list([{
        'text': f'example {index}'
    } for index in range(4)])

    result = AveragePPLEvaluator().evaluate(
        k=1,
        n=2,
        original_dataset=dataset,
        ppl=[1.0, 3.0, 10.0, 20.0],
    )

    assert result == {'average_ppl (2 runs average)': 8.5}
