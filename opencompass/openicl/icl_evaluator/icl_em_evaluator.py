from opencompass.registry import ICL_EVALUATORS
from opencompass.utils.text_postprocessors import general_postprocess

from .icl_base_evaluator import BaseEvaluator


@ICL_EVALUATORS.register_module()
class EMEvaluator(BaseEvaluator):
    """Exact match evaluator."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        predictions = [
            general_postprocess(prediction) for prediction in predictions
        ]
        processed_answers = [[general_postprocess(j) for j in i]
                             for i in references]

        cnt = 0
        details = []
        for pred, ans, origin_ans in zip(predictions, processed_answers,
                                         references):
            answers = list(set(ans + origin_ans))
            detail = {'pred': pred, 'answer': answers}
            if pred in ans or pred in origin_ans:
                cnt += 1
                detail['correct'] = True
            else:
                detail['correct'] = False
            details.append(detail)

        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details}
