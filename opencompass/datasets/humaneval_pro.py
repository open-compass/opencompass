import json
from typing import Dict, List

import numpy as np
from datasets import Dataset

from opencompass.openicl.icl_evaluator.code_evaluator import CodeEvaluator
from opencompass.utils import get_data_path

from .base import BaseDataset


class HumanevalevalProDataset(BaseDataset):

    @staticmethod
    def load(path, num_repeats=1, local_mode=False):
        path = get_data_path(path, local_mode=local_mode)
        dataset = []
        with open(path, encoding='utf-8') as f:
            raw_data = json.load(f)
            for data in raw_data:
                dataset.extend([data for _ in range(num_repeats)])
        return Dataset.from_list(dataset)


class HumanevalProEvaluator(CodeEvaluator):

    def _process_completions(self, test_case: dict, completions: list) -> list:
        processed_completions = []
        for comp in completions:
            post_comp = self._extract_code(comp)
            processed_completions.append(post_comp)
        return processed_completions

    def score(self, predictions: List, references: List,
              test_set: Dataset) -> Dict:
        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }

        test_set = test_set.to_pandas()
        # Use the first column as the unique identifier
        test_set_origin = test_set.drop_duplicates(subset=test_set.columns[0])
        num_repeats = int(len(test_set) / len(test_set_origin))

        # 1. Prepare data for all test cases
        all_test_cases = []
        for i in range(len(test_set_origin)):
            test_case = test_set_origin.iloc[i]
            completions = predictions[i * num_repeats:(i + 1) * num_repeats]

            # Process code completions
            processed_completions = self._process_completions(
                test_case, completions)

            sub_data_dict = {
                'name': int(test_case['id']),
                'language': self.language,
                'prompt': '',
                'tests': test_case['test_code'],
                'processed_completions': processed_completions,
                'completions': completions
            }

            all_test_cases.append(sub_data_dict)

        # 2. Send all test cases to the evaluation service
        success, outputs, error_message = self._evaluate(all_test_cases)
        if not success:
            return {'error': error_message}

        # 3. Process the returned results
        details = []
        total, correct = [], []
        for output in outputs:
            passed = [m['status'] == 'OK' for m in output['meta_data']]
            total.append(len(passed))
            correct.append(sum(passed))
            details.append(output)
        total = np.array(total)
        correct = np.array(correct)

        pass_at_k = {
            f'pass@{k}':
            self.estimate_pass_at_k(total, correct, k).mean() * 100
            for k in self.k if (total >= k).all()
        }

        return {
            **pass_at_k,
            'details': details,
        }
