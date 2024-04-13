import re

from datasets import load_dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS, ICL_EVALUATORS

from opencompass.openicl.icl_evaluator import BaseEvaluator
from ..base import BaseDataset
from . import utils
from tqdm import tqdm


@LOAD_DATASET.register_module()
class TheoremQADatasetV3(BaseDataset):
    @staticmethod
    def load(path: str):
        return load_dataset("csv", data_files={"test": path})


def TheoremQA_postprocess_v3(text: str) -> str:
    answer = utils.answer_clean(["The answer is:", "The answer is", "the answer is"], text)
    return answer


@ICL_EVALUATORS.register_module()
class TheoremQAEvaluatorV3(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {"error": "preds and refrs have different length"}

        details = []
        correct, wrong = 0, 0
        for answer, groundtruth in zip(tqdm(predictions), references):
            if isinstance(groundtruth, str):
                groundtruth = [groundtruth]
            if utils.compare_answer_with_groundtruth(answer, *groundtruth):
                correct += 1
                is_correct = True
            else:
                wrong += 1
                is_correct = False

            details.append(
                {
                    # "question": question,
                    # "solution": output,
                    "correct": groundtruth,
                    "pred": answer,
                    "is_correct": is_correct,
                }
            )

        score = correct / (correct + wrong) * 100
        return {'score': score, 'details': details}
