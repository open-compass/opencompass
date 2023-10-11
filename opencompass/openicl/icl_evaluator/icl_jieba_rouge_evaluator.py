import jieba
from rouge_chinese import Rouge

from opencompass.registry import ICL_EVALUATORS
from opencompass.utils.text_postprocessors import general_postprocess

from .icl_base_evaluator import BaseEvaluator


@ICL_EVALUATORS.register_module()
class JiebaRougeEvaluator(BaseEvaluator):
    """This Evaluator will first use jieba for tokenization, and then calculate
    the rouge score.

    This Evaluator especially suitable for evaluating Chinese datasets.
    """

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        predictions = [general_postprocess(i) for i in predictions]
        references = [general_postprocess(i) for i in references]

        metric = Rouge()
        predictions = [' '.join(jieba.cut(i)) for i in predictions]
        references = [' '.join(jieba.cut(i)) for i in references]

        # avoid raising error when empty string encountered
        predictions = [i if i else '__PREDPLACEHOLDER__' for i in predictions]
        references = [i if i else '__REFRPLACEHOLDER__' for i in references]

        score = metric.get_scores(predictions, references, avg=True)

        return {
            'rouge1': score['rouge-1']['f'] * 100,
            'rouge2': score['rouge-2']['f'] * 100,
            'rougeL': score['rouge-l']['f'] * 100,
        }
