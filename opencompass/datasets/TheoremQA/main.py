import re
import json

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS, ICL_EVALUATORS
from opencompass.utils import get_data_path

from opencompass.openicl.icl_evaluator import BaseEvaluator
from ..base import BaseDataset
from . import utils
from tqdm import tqdm


@LOAD_DATASET.register_module()
class TheoremQADatasetV3(BaseDataset):

    @staticmethod
    def load(path: str):
        path = get_data_path(path, local_mode=True)
        with open(path, 'r') as f:
            data = json.load(f)
        for item in data:
            item['Answer'] = str(item['Answer'])
        dataset = Dataset.from_list(data)
        return dataset


def TheoremQA_postprocess_v3(text: str) -> str:
    answer = utils.answer_clean(["The answer is:", "The answer is", "the answer is"], text)
    return answer


@ICL_EVALUATORS.register_module()
class TheoremQAEvaluatorV3(BaseEvaluator):
    def score(self, predictions, references, test_set):
        if len(predictions) != len(references):
            return {"error": "preds and refrs have different length"}

        details = []
        correct, wrong = 0, 0
        for index in tqdm(range(len(predictions))):
            answer = predictions[index]
            groundtruth = references[index]
            answer_type = test_set[index]['Answer_type']
            if answer_type in ['float', 'integer', 'bool']:
                groundtruth = [groundtruth, eval(groundtruth)]
            else:
                groundtruth = [groundtruth, None]
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
