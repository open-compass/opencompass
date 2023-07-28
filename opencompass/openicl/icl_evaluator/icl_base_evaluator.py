"""Base Evaluator."""


class BaseEvaluator:

    def __init__(self) -> None:
        pass

    def score(self):
        raise NotImplementedError("Method hasn't been implemented yet")
