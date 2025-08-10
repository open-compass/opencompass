from typing import List

import numpy as np

from opencompass.registry import ICL_EVALUATORS

from .icl_base_evaluator import BaseEvaluator


@ICL_EVALUATORS.register_module()
class BPCEvaluator(BaseEvaluator):

    def score(self, loss: List[float], total_chr_num: List[float]):
        """Calculate bits per character based on inference results.

        Args:
            loss (List[float]): CrossEntropyLoss per batch x sliding
            context window
            total_chr_num (List[float]): Total number of characters
            in the original dataset.

        Returns:
            Dict[str, float]: Bits per Character
        """
        total_loss = sum(loss)

        # Multiplying by log(2) to correct for the constant shift
        # due to natural log used in the PyTorch implementation
        # of CrossEntropyLoss
        bpc = total_loss / (total_chr_num[0] * np.log(2))

        return {'bpc': bpc}
