from opencompass.openicl import BaseEvaluator


def check(a, b):
    return abs(float(a) - float(b)) < 1e-3


class Math401Evaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            try:
                if check(i, j):
                    correct += 1
                    detail['correct'] = True
            except Exception:
                pass
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result
