# flake8: noqa: E501

import difflib
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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
                 language: str,
                 ip_address: str = 'localhost',
                 retry: int = 3) -> None:
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
            result = self.client.predict(input_data, api_name='/evaluate')

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

    def _remove_prefix(self,
                       prompt: str,
                       completion: str,
                       threshold: float = 0.95) -> str:
        """Determine the truncation point in the completion based on the last
        line of the prompt, remove all content before that line in the
        completion, and return the completion string after removing the prefix.
        This is done to convert chatbot-style inference mode to completion
        mode.

        Args:
            prompt (str): The prompt text.
            completion (str): The completion text.
            threshold (float): Line similarity threshold.

        Returns:
            str: The completion string after removing the prefix.
        """
        prompt_lines = prompt.splitlines()
        completion_lines = completion.splitlines()

        if not prompt_lines:
            return completion

        last_prompt_line = prompt_lines[-1]
        cut_index = -1

        for i, completion_line in enumerate(completion_lines):
            similarity = difflib.SequenceMatcher(None, last_prompt_line,
                                                 completion_line).ratio()
            if similarity >= threshold:
                cut_index = i
                break

        if cut_index != -1:
            return '\n'.join(completion_lines[cut_index + 1:])
        else:
            return completion

    def _process_completions(self, test_case: dict, completions: list) -> list:
        """Process code completion list, which typically involves extracting
        code, removing repetitive prefixes caused by chatbot mode, and other
        steps to ensure the model-generated code can be compiled successfully.

        Args:
            test_case (dict): Dictionary containing test case information including:
            completions (list): List of code completions generated by the model.

        Returns:
            list: Processed code completion list.
        """
        processed_completions = []
        for comp in completions:
            comp = self._extract_code(comp)
            post_comp = self._remove_prefix(test_case['prompt'], comp)
            processed_completions.append(post_comp)
        return processed_completions

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
                time.sleep(10)
            else:
                break

        if not succeed:
            return False, None, f'code eval service connection failed: {output}'

        return True, output, None

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
        num_repeats = int(len(test_set) / len(test_set_origin))

        # 1. Prepare data for all test cases
        all_test_cases = []
        for i in range(len(test_set_origin)):
            test_case = test_set_origin.iloc[i]
            completions = predictions[i * num_repeats:(i + 1) * num_repeats]

            # Process code completions
            processed_completions = self._process_completions(
                test_case, completions)

            result_dict = {
                'name': test_case['name'],
                'language': test_case['language'],
                'prompt': test_case['prompt'],
                'tests': test_case['tests'],
                'processed_completions': processed_completions,
                'completions': completions
            }

            all_test_cases.append(result_dict)

        # 2. Send all test cases to the evaluation service
        success, outputs, error_message = self._evaluate(all_test_cases)
        if not success:
            return {'error': error_message}

        # 3. Process the returned results
        details = []
        correct = 0
        for output in outputs:
            if output.get('status') == 'OK':
                output['correct'] = True
                correct += 1
            else:
                output['correct'] = False

            details.append(output)

        return {
            f'pass@{num_repeats}': 100 * correct / len(test_set_origin),
            'details': details
        }
