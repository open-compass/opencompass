from opencompass.registry import ICL_EVALUATORS

from .icl_base_evaluator import BaseEvaluator


@ICL_EVALUATORS.register_module()
class AveragePPLEvaluator(BaseEvaluator):

    def score(self, ppl):
        average_ppl = sum(ppl) / len(ppl)
        return {'average_ppl': average_ppl}
