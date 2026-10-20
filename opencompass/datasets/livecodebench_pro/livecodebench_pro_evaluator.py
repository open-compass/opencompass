# flake8: noqa: E501

import re
import time
from typing import Dict, List

import requests
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


def extract_longest_cpp_code(text):
    """Extract C++ code from text.

    First tries to find fenced code blocks with ```cpp.
    If not found, looks for code containing main() function and #include.

    Args:
        text (str): Text containing C++ code

    Returns:
        str or None: Extracted C++ code or None if not found
    """
    # -------------------------------
    # 1. First match all fenced code blocks starting with ```cpp at the beginning of a line
    # -------------------------------
    fenced_pattern = r'(?m)^```cpp\s*\n(.*?)\n```'
    fenced_blocks = re.findall(fenced_pattern, text, flags=re.DOTALL)
    if fenced_blocks:
        # Search from the last one backwards, return the first block containing "#include"
        for block in reversed(fenced_blocks):
            if '#include' in block:
                return block.strip()
    # -------------------------------
    # 2. If no suitable fenced code blocks are found, extract code blocks based on main occurrence position
    #    Start from the last main and work backwards, only return if conditions are met
    # -------------------------------
    cleaned_text = text  # Keep original text unchanged
    main_matches = list(re.finditer(r'int\s+main\s*\(', cleaned_text))
    if main_matches:
        # Traverse backwards from the last main
        for main in reversed(main_matches):
            main_start_pos = main.start()
            main_end_pos = main.end()

            # From main_end_pos, find the start of main's internal code block: the first left brace '{'
            brace_start = cleaned_text.find('{', main_end_pos)
            if brace_start == -1:
                # Cannot find left brace, skip this main
                continue
            # Brace matching, find the corresponding right brace '}' until count returns to zero
            brace_count = 0
            idx = brace_start
            text_len = len(cleaned_text)
            while idx < text_len:
                ch = cleaned_text[idx]
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        idx += 1  # Include the closing brace
                        break
                idx += 1
            func_end = idx  # End position of main function block
            # Split text by lines and record the start index of each line in the original text (assuming newline length is 1)
            lines = cleaned_text.splitlines()
            line_start_indices = []
            curr_idx = 0
            for line in lines:
                line_start_indices.append(curr_idx)
                curr_idx += len(line) + 1

            # Locate the line where main appears
            main_line_index = None
            for i, start in enumerate(line_start_indices):
                if start <= main_start_pos < (start + len(lines[i]) + 1):
                    main_line_index = i
                    break
            if main_line_index is None:
                main_line_index = 0

            # Scan upwards for consecutive "#include" lines (including consecutive #include lines above the main line)
            include_line_index = None
            for i in range(main_line_index, -1, -1):
                if re.match(r'^\s*#include', lines[i]):
                    include_line_index = i
                else:
                    # Once a non-#include line is encountered and #include lines have been found before, stop scanning
                    if include_line_index is not None:
                        break

            candidate_start = (line_start_indices[include_line_index]
                               if include_line_index is not None else
                               line_start_indices[main_line_index])

            candidate_code = cleaned_text[candidate_start:func_end].strip()
            if '#include' in candidate_code:
                return candidate_code

    return None


def extract_longest_python_code(text):
    """Extract Python code from text.

    First tries to find fenced code blocks with ```python.
    If not found, looks for function/class definitions or import statements.

    Args:
        text (str): Text containing Python code

    Returns:
        str or None: Extracted Python code or None if not found
    """
    # -------------------------------
    # 1. First match all fenced code blocks starting with ```python
    # -------------------------------
    fenced_pattern = r'(?m)^```python\s*\n(.*?)\n```'
    fenced_blocks = re.findall(fenced_pattern, text, flags=re.DOTALL)

    if fenced_blocks:
        # Return the longest Python code block
        longest_block = max(fenced_blocks, key=len)
        return longest_block.strip()

    # -------------------------------
    # 2. If no fenced blocks, look for Python function/class definitions
    # -------------------------------

    # Pattern to match Python function/class definitions
    def_class_pattern = r'(def\s+\w+|class\s+\w+)'
    def_class_matches = list(re.finditer(def_class_pattern, text))

    if def_class_matches:
        # Try to extract code blocks containing these definitions
        code_blocks = []

        for match in def_class_matches:
            start_pos = match.start()

            # Find the boundaries of this code block
            # Look for the beginning (previous blank line or start of text)
            block_start = 0
            for i in range(start_pos, -1, -1):
                if i == 0:
                    block_start = i
                    break
                # Check if encountering blank line (two consecutive newlines)
                elif i >= 1 and text[i - 1:i + 1] == '\n\n':
                    block_start = i + 1  # Start after blank line
                    break

            # Look for the end (next blank line or end of text)
            block_end = len(text)
            for i in range(start_pos, len(text)):
                if i == len(text) - 1:
                    block_end = len(text)
                    break
                # Check if encountering blank line
                elif i < len(text) - 1 and text[i:i + 2] == '\n\n':
                    block_end = i + 1  # Include the first newline
                    break

            code_block = text[block_start:block_end].strip()
            if code_block and ('def ' in code_block or 'class ' in code_block):
                code_blocks.append(code_block)

        if code_blocks:
            # Return the longest code block containing a function/class definition
            return max(code_blocks, key=len)

    # -------------------------------
    # 3. If no function/class definitions, look for import statements
    # -------------------------------
    import_pattern = r'(^import\s+\w+|^from\s+\w+\s+import)'
    import_matches = list(re.finditer(import_pattern, text,
                                      flags=re.MULTILINE))

    if import_matches:
        # Extract code around import statements
        import_blocks = []

        for match in import_matches:
            start_pos = match.start()

            # Find boundaries
            block_start = 0
            for i in range(start_pos, -1, -1):
                if i == 0:
                    block_start = i
                    break
                elif i >= 1 and text[i - 1:i + 1] == '\n\n':
                    block_start = i + 1
                    break

            block_end = len(text)
            for i in range(start_pos, len(text)):
                if i == len(text) - 1:
                    block_end = len(text)
                    break
                elif i < len(text) - 1 and text[i:i + 2] == '\n\n':
                    block_end = i + 1
                    break

            import_block = text[block_start:block_end].strip()
            if import_block and ('import ' in import_block
                                 or 'from ' in import_block):
                import_blocks.append(import_block)

        if import_blocks:
            return max(import_blocks, key=len)

    return None


@ICL_EVALUATORS.register_module()
class LCBProEvaluator(BaseEvaluator):
    """Evaluator for LiveCodeBench Pro dataset.

    This evaluator extracts code from model outputs (Python or C++),
    submits them to a remote evaluation service, and polls for results.

    Args:
        submit_url (str): URL for submitting code. Defaults to the LCB Pro service.
        result_url (str): URL template for retrieving results. Defaults to the LCB Pro service.
        timeout (int): Request timeout in seconds. Defaults to 10.
        poll_interval (int): Interval between result polling in seconds. Defaults to 10.
        max_retries (int): Maximum number of retries for failed requests. Defaults to 3.
    """

    def __init__(
        self,
        submit_url: str = 'http://lightcpverifier.ailab.ailab.ai/submit',
        result_url:
        str = 'http://lightcpverifier.ailab.ailab.ai/result/{submission_id}',
        timeout: int = 10,
        poll_interval: int = 10,
        max_retries: int = 3,
    ) -> None:
        """Initialize the LCBProEvaluator."""
        self.submit_url = submit_url
        self.result_url = result_url
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        super().__init__()

    def _submit_code(self, pid: str, lang: str, code: str) -> int:
        """Submit code to the evaluation service.

        Args:
            pid (str): Problem ID
            lang (str): Programming language ('python' or 'cpp')
            code (str): Code to evaluate

        Returns:
            int: Submission ID

        Raises:
            Exception: If submission fails after retries
        """
        payload = {
            'pid': pid,
            'lang': lang,
            'code': code,
        }
        no_proxy = {'http': None, 'https': None}

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.submit_url,
                                         json=payload,
                                         timeout=self.timeout,
                                         proxies=no_proxy)
                response.raise_for_status()
                return response.json()['sid']
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(
                        f'Failed to submit code after {self.max_retries} attempts: {e}'
                    )
                time.sleep(1)

        raise Exception('Should not reach here')

    def _get_result(self, submission_id: int) -> str:
        """Get evaluation result for a submission.

        Args:
            submission_id (int): Submission ID

        Returns:
            str: Result status ('Judging', 'Accepted', 'Judge Failed', etc.)
        """
        url = self.result_url.format(submission_id=submission_id)
        no_proxy = {'http': None, 'https': None}

        try:
            response = requests.get(url,
                                    proxies=no_proxy,
                                    timeout=self.timeout)
            if response.status_code == 404:
                return 'Judging'
            response.raise_for_status()
            info = response.json()
            status = info.get('status', '')
            if status in ('queued', 'running', 'pending'):
                return 'Judging'
            if status == 'error':
                return 'Judge Failed'
            return info.get('result', 'Unknown')
        except Exception as e:
            return f'Error: {e}'

    def _extract_code(self, text: str) -> tuple:
        """Extract code from model output.

        Tries to extract C++ code first, then Python code.

        Args:
            text (str): Model output text

        Returns:
            tuple: (code, language) or (None, None) if no code found
        """
        # Try C++ first
        if re.search(r'```cpp', text):
            code = extract_longest_cpp_code(text)
            if code is not None:
                return code, 'cpp'

        # Try Python
        if re.search(r'```python', text):
            code = extract_longest_python_code(text)
            if code is not None:
                return code, 'python'

        # If no language marker, try both extractors
        cpp_code = extract_longest_cpp_code(text)
        if cpp_code is not None:
            return cpp_code, 'cpp'

        python_code = extract_longest_python_code(text)
        if python_code is not None:
            return python_code, 'python'

        return None, None

    def score(self, predictions: List, references: List,
              test_set: Dataset) -> Dict:
        """Score code generation predictions against references.

        Args:
            predictions (list): List of model-generated code completions.
            references (list): List of reference problem IDs.
            test_set (Dataset): Dataset containing problem information.

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

        # Convert dataset to pandas for easier manipulation
        test_set = test_set.to_pandas()

        # Track submissions
        submissions = []
        details = []

        # Step 1: Extract code and submit
        from tqdm import tqdm
        print('Submitting code to evaluation service...')
        for i in tqdm(range(len(predictions))):
            prediction = predictions[i]
            problem_id = references[i]

            # Extract code from prediction
            code, lang = self._extract_code(prediction)

            if code is None:
                # No code found
                submissions.append({
                    'index': i,
                    'problem_id': problem_id,
                    'sid': None,
                    'code': None,
                    'lang': None,
                    'error': 'No code extracted'
                })
            else:
                try:
                    # Submit code
                    sid = self._submit_code(problem_id, lang, code)
                    submissions.append({
                        'index': i,
                        'problem_id': problem_id,
                        'sid': sid,
                        'code': code,
                        'lang': lang,
                        'error': None
                    })
                except Exception as e:
                    submissions.append({
                        'index': i,
                        'problem_id': problem_id,
                        'sid': None,
                        'code': code,
                        'lang': lang,
                        'error': str(e)
                    })

        # Step 2: Poll for results
        print('Polling for evaluation results...')
        total_count = len(submissions)
        accepted_count = 0

        for submission in tqdm(submissions):
            if submission['sid'] is None:
                # Submission failed
                details.append({
                    'problem_id':
                    submission['problem_id'],
                    'correct':
                    False,
                    'result':
                    submission.get('error', 'Unknown error'),
                    'code':
                    submission.get('code'),
                    'lang':
                    submission.get('lang'),
                })
                continue

            # Poll for result
            sid = submission['sid']
            while True:
                result = self._get_result(sid)
                if result != 'Judging':
                    if 'Accepted' in result:
                        accepted_count += 1
                        details.append({
                            'problem_id': submission['problem_id'],
                            'correct': True,
                            'result': result,
                            'code': submission['code'],
                            'lang': submission['lang'],
                        })
                    else:
                        details.append({
                            'problem_id': submission['problem_id'],
                            'correct': False,
                            'result': result,
                            'code': submission['code'],
                            'lang': submission['lang'],
                        })
                    break
                time.sleep(self.poll_interval)

        # Calculate accuracy
        accuracy = 100 * accepted_count / total_count if total_count > 0 else 0

        return {
            'accuracy': accuracy,
            'pass@1': accuracy,  # Alias for consistency with other evaluators
            'details': details,
        }
