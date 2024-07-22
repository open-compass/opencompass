from opencompass.registry import ICL_EVALUATORS

from .icl_base_evaluator import BaseEvaluator


@ICL_EVALUATORS.register_module()
class AveragePPLEvaluator(BaseEvaluator):

    def score(self, ppl):
        average_ppl = sum(ppl) / len(ppl)
        return {'average_ppl': average_ppl}


@ICL_EVALUATORS.register_module()
class AverageMinKEvaluator(BaseEvaluator):

    def score(self, mink):
        average_mink = sum(mink) / len(mink)
        return {'average_mink': average_mink}


@ICL_EVALUATORS.register_module()
class AverageInferencePPLEvaluator(BaseEvaluator):

    def score(self, ppl, token_len):
        average_ppl = sum(ppl) / sum(token_len)
        return {'average_ppl': average_ppl}
