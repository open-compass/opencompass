from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score

from opencompass.registry import ICL_EVALUATORS

from .icl_base_evaluator import BaseEvaluator


@ICL_EVALUATORS.register_module()
class AUCROCEvaluator(BaseEvaluator):
    """Calculate AUC-ROC scores and accuracy according the prediction.

    For some dataset, the accuracy cannot reveal the difference between
    models because of the saturation. AUC-ROC scores can further exam
    model abilities to distinguish different labels. More details can refer to
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    """  # noqa

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions: List, references: List) -> dict:
        """Calculate scores and accuracy.

        Args:
            predictions (List): List of probabilities for each class of each
                sample.
            references (List): List of target labels for each sample.

        Returns:
            dict: calculated scores.
        """
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length.'
            }
        auc_score = roc_auc_score(references, np.array(predictions)[:, 1])
        accuracy = sum(
            references == np.argmax(predictions, axis=1)) / len(references)
        return dict(auc_score=auc_score * 100, accuracy=accuracy * 100)
