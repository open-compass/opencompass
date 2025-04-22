from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class InternSandboxDataset(BaseDataset):

    @staticmethod
    def load(path: str, local_mode: bool = False):
        pass


@ICL_EVALUATORS.register_module()
class MBPPEvaluator(BaseEvaluator):

    def score(self, predictions, references, test_set):
        pass
