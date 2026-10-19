# flake8: noqa: E501

import json
from typing import Dict, List

from datasets import Dataset

from opencompass.openicl.icl_evaluator.code_evaluator import CodeEvaluator
from opencompass.utils import get_data_path

from .base import BaseDataset

PROMPT_WRAPPER = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
Write a solution of python file to the following problems, the solution of the second problem requires single or multiple calls to the first solution.
```python
{raw_problem}
{new_problem}
```
Please put the two solutions within the Python code block provided below, and make sure that the block contains no other unrelated content:
```python
```
"""


class MBPPProDataset(BaseDataset):

    @staticmethod
    def load(path, local_mode=False):
        path = get_data_path(path, local_mode=local_mode)
        print(path)
        dataset = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
        return Dataset.from_list(dataset)


class MBPPProEvaluator(CodeEvaluator):

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

        # 1. Prepare data for all test cases
        all_test_cases, prompts = [], []
        for i in range(len(test_set_origin)):
            test_case = test_set_origin.iloc[i]
            completion = predictions[i]

            # Process code completions
            processed_completion = self._process_completions(completion)
            code = processed_completion + '\n' + test_case['test_code']
            sub_data_dict = {
                'name': int(test_case['id']),
                'language': self.language,
                'code': code,
            }
            all_test_cases.append(sub_data_dict)

            prompt = PROMPT_WRAPPER.format(
                raw_problem=test_case['raw_problem'],
                new_problem=test_case['new_problem'])
            prompts.append(prompt)

        # 2. Send all test cases to the evaluation service
        success, outputs, error_message = self._evaluate(all_test_cases)
        if not success:
            return {'error': error_message}

        # 3. Process the returned results
        return self._process_results(outputs, prompts, len(test_set_origin))
