"""Base Evaluator."""


class BaseEvaluator:

    def __init__(self) -> None:
        pass

    def score(self):
        raise NotImplementedError("Method hasn't been implemented yet")

    @staticmethod
    def is_num_equal(predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        else:
            return
