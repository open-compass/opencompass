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
