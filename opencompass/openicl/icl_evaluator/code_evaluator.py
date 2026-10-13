# flake8: noqa: E501

import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from datasets import Dataset
from gradio_client import Client

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


@ICL_EVALUATORS.register_module()
class CodeEvaluator(BaseEvaluator):
    """Evaluator for code generation tasks.

    This evaluator sends code to a remote evaluation service to test its
    functionality against provided test cases. It handles code extraction,
    processing, and result analysis.
    """

    def __init__(self,
                 language: str = 'py',
                 ip_address: str = 'localhost',
                 retry: int = 5) -> None:
        """Initialize the CodeEvaluator.

        Args:
            language (str): Programming language of the code to evaluate.
            ip_address (str, optional): IP address of the evaluation service. Defaults to 'localhost'.
            retry (int, optional): Number of retry attempts for failed connections. Defaults to 3.
        """
        self.language = language
        self.retry = retry
        self.client = Client(ip_address)
        super().__init__()

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown-formatted text.

        Args:
            text (str): Text that may contain code blocks in markdown format.

        Returns:
            str: Extracted code from the last code block, or the original text if no code blocks found.
        """
        blocks = re.findall(r'```\w*\n(.*?)```', text, re.DOTALL)
        if len(blocks) >= 1:
            text = blocks[0]
        return text

    def _code_eval_service(
        self, input_data: Union[Dict, List,
                                str]) -> Tuple[bool, Union[Dict, List, Any]]:
        """Send code to the remote evaluation service using gradio_client and
        get the results.

        Args:
            input_data: Can be one of:
                - dict: Dictionary containing code information for a single test case
                - list: List of dictionaries for batch evaluation
                - str: File path to code file

        Returns:
            tuple: (succeed, output)
                - succeed (bool): Whether the request was successful
                - output (dict/list/str): Evaluation results or error message
        """
        try:
            import requests
            temp_file_path = None
            # Handle file path input
            if isinstance(input_data, str):
                with tempfile.NamedTemporaryFile(suffix=f'.{self.language}',
                                                 delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    with open(input_data, 'r') as src_file:
                        content = src_file.read()
                    temp_file.write(content.encode())
                input_data = temp_file_path

            # Send to evaluation service
            try:
                result = self.client.predict(input_data, api_name='/evaluate')
            except Exception as e:
                # Catch timeout and other exceptions
                if 'timed out' in str(e).lower() or 'timeout' in str(
                        e).lower():
                    return False, f'Request to code eval service timed out: {e}'
                else:
                    raise

            # Process the result
            if isinstance(result, (dict, list)):
                return True, result
            else:
                # Try to parse the result as JSON if it's a string
                try:
                    import json
                    parsed_result = json.loads(result)
                    return True, parsed_result
                except:  # noqa: E722
                    return True, {'status': 'unknown', 'raw_result': result}

        except Exception as e:
            return False, str(e)
        finally:
            # Clean up temporary file if it was created
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:  # noqa: E722
                    pass

    def _process_completions(self, completion: str) -> list:
        """Process code completions to extract the relevant code.

        Args:
            completion (str): Code completion string.
        Returns:
            list: List of processed code completions.
        """
        post_comp = self._extract_code(completion)
        return post_comp

    def _evaluate(
        self, input_data: Union[Dict, List]
    ) -> Tuple[bool, Optional[Union[Dict, List]], Optional[str]]:
        """Evaluate code with retry mechanism.

        Args:
            input_data: Can be either:
                - dict: Dictionary containing code and test information for a single test case
                - list: List of dictionaries for batch evaluation

        Returns:
            tuple: (success, output, error_message)
                - success (bool): Whether the evaluation was successful
                - output (dict or list): Evaluation output (if successful)
                - error_message (str): Error message (if failed)
        """
        num_retry = 0
        while num_retry < self.retry:
            succeed, output = self._code_eval_service(input_data)
            if not succeed:
                num_retry += 1
                time.sleep(30)
            else:
                break

        if not succeed:
            return False, None, f'code eval service connection failed: {output}'

        return True, output, None

    def _process_results(self, outputs: List, prompts: List,
                         total_count: int) -> Dict:
        """Process the evaluation results.
        Args:
            outputs (list): List of evaluation results for each test case.
            prompts (list): List of prompts used for each test case.
            total_count (int): Total number of test cases.
        Returns:
            dict: Processed results including:
                - pass@1: Percentage of test cases passed
                - details: Detailed results for each test case
        """
        details = []
        correct = 0
        for output, prompt in zip(outputs, prompts):
            output['prompt'] = prompt
            if output.get('status') == 'OK':
                output['correct'] = True
                correct += 1
            else:
                output['correct'] = False
            details.append(output)

        return {f'pass@1': 100 * correct / total_count, 'details': details}

    def score(self, predictions: List, references: List,
              test_set: Dataset) -> Dict:
        """Score code generation predictions against references.

        Args:
            predictions (list): List of model-generated code completions.
            references (list): List of reference solutions (not directly used in evaluation).
            test_set (Dataset): Dataset containing test cases and other metadata.

        Returns:
            dict: Evaluation results including:
                - accuracy: Percentage of correctly solved problems
                - details: Detailed results for each test case
                - error: Error message if evaluation failed
        """
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
            processed_completion = self._process_completions(
                test_case, completion)
            code = test_case[
                'prompt'] + processed_completion + '\n' + test_case['tests']
            sub_data_dict = {
                'name': test_case['name'],
                'language': test_case['language'],
                'code': code
            }
            all_test_cases.append(sub_data_dict)
            prompts.append(test_case['prompt'])

        # 2. Send all test cases to the evaluation service
        success, outputs, error_message = self._evaluate(all_test_cases)
        if not success:
            return {'error': error_message}

        # 3. Process the returned results
        return self._process_results(outputs, prompts, len(test_set_origin))
