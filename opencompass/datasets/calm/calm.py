from typing import List

import datasets
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset
from .data_processing.generate_questions import generate_question_list
from .evaluation.core_metrics import compute_core_metrics
from .evaluation.errors import identify_model_errors


@LOAD_DATASET.register_module()
class CaLMDataset(BaseDataset):

    @staticmethod
    def load(path: str, prompt_style: str) -> datasets.Dataset:
        question_list = generate_question_list(dataset_path=path,
                                               prompt_style=prompt_style)
        dataset = Dataset.from_list(question_list)
        return dataset


class CaLMEvaluator(BaseEvaluator):

    def __init__(self, core_metrics, error_analysis, prompt_style,
                 task) -> None:
        super().__init__()
        self.core_metrics = core_metrics
        self.error_analysis = error_analysis
        self.prompt_style = prompt_style
        self.task = task

    def score(
        self,
        predictions: List,
        references: List,
    ) -> dict:
        results = {}
        if self.core_metrics:
            metrics, pred_list = compute_core_metrics(
                predictions,
                task=self.task,
                prompt_style=self.prompt_style,
                gt_items=references)
            results.update(metrics)
        if self.error_analysis:
            if self.task.startswith('CEG-O_E-CARE'):
                print("There's no error analysis for CEG-O_E-CARE task. ",
                      'Skipping error analysis.')
                return results
            errors = identify_model_errors(
                predictions,
                task=self.task,
                prompt_style=self.prompt_style,
                gt_items=references)  # Define specific criteria
            results.update(errors)
        return results
