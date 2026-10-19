import json
import os

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .evaluation_functions import (cjft, flzx, ftcs, jdzy, jec_ac, jec_kd,
                                   jetq, lblj, ljp_accusation, ljp_article,
                                   ljp_imprison, sjjc, wbfl, wsjd, xxcq, ydlj,
                                   yqzy, zxfl)


@LOAD_DATASET.register_module()
class LawBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, index: str) -> Dataset:
        path = get_data_path(path, local_mode=True)
        path = os.path.join(path, index + '.json')
        with open(path, 'r') as f:
            data = json.load(f)
        return Dataset.from_list(data)


funct_dict = {
    '1-1': ftcs.compute_ftcs,
    '1-2': jec_kd.compute_jec_kd,
    '2-1': wsjd.compute_wsjd,
    '2-2': jdzy.compute_jdzy,
    '2-3': wbfl.compute_wbfl,
    '2-4': zxfl.compute_zxfl,
    '2-5': ydlj.compute_ydlj,
    '2-6': xxcq.compute_xxcq,
    '2-7': yqzy.compute_yqzy,
    '2-8': lblj.compute_lblj,
    '2-9': sjjc.compute_sjjc,
    '2-10': sjjc.compute_cfcy,
    '3-1': ljp_article.compute_ljp_article,
    '3-2': cjft.compute_cjft,
    '3-3': ljp_accusation.compute_ljp_accusation,
    '3-4': ljp_imprison.compute_ljp_imprison,
    '3-5': ljp_imprison.compute_ljp_imprison,
    '3-6': jec_ac.compute_jec_ac,
    '3-7': jetq.compute_jetq,
    '3-8': flzx.compute_flzx,
}


class LawBenchEvaluator(BaseEvaluator):

    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def score(self, predictions, references, origin_prompt):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        data_dict = [{
            'origin_prompt': origin_prompt[i],
            'prediction': predictions[i],
            'refr': references[i],
        } for i in range(len(predictions))]
        scores = funct_dict[self.index](data_dict)
        scores = {k: v * 100 for k, v in scores.items()}

        return scores


for index in funct_dict:
    # fix classic closure problem
    def _register(index):
        ICL_EVALUATORS.register_module(
            name='LawBenchEvaluator_' + index.replace('-', '_'),
            module=lambda *args, **kwargs: LawBenchEvaluator(
                index=index, *args, **kwargs))

    _register(index)
