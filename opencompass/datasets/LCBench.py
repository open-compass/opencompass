# flake8: noqa
import contextlib
import io
import itertools
import multiprocessing
import re
import signal
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Sequence, Union

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class LCDataset(BaseDataset):

    @staticmethod
    def load(path: str,
             num_repeats: int = 1,
             difficulty='ALL',
             local_mode=False):
        """Load LC dataset for pass k mode.

        Note that you can use num_repeats > 1 when your model does not support
        `num_return_sequence` in generation, otherwise use the raw
        LC dataset and set `num_return_sequence` in model config to
        generate multiple responses for testing pass@k>1.

        It better to change your dataset abbr correspondingly if you want to
        change num_repeats>1, otherwise the number in
        `.cache/dataset_size.json` might be inconsistent.

        Args:
            num_repeats(int): Number of repetition for this dataset to get
        multiple responses in special cases.
        """
        path = get_data_path(path, local_mode=local_mode)

        def processing_test(example):
            example['test_case'] = example['test_list']
            example['test_list'] = '\n'.join(example['test_list'])
            example['test_column'] = dict(test_list_2=example['test_list'],
                                          task_id=example['Contest id'])
            return example

        train = load_dataset('json', data_files=path,
                             split='train[:5]').map(processing_test)
        test = load_dataset('json', data_files=path,
                            split='train[5:]').map(processing_test)
        if not difficulty == 'ALL':
            train = train.filter(
                lambda example: example['Difficulty'] == difficulty)
            test = test.filter(
                lambda example: example['Difficulty'] == difficulty)
        test = concatenate_datasets([test] * num_repeats)
        return DatasetDict({'train': train, 'test': test})


class TimeOutException(Exception):
    pass


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):

    def signal_handler(signum, frame):
        raise TimeOutException('Time out!')

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@ICL_EVALUATORS.register_module()
class LCEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        result = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
        details = {}

        with ProcessPoolExecutor() as executor:
            futures = []
            for i, (refer, pred) in enumerate(zip(references, predictions)):

                code_blocks = self._process_answer(pred)

                # Try each code block until one passes
                for code_idx, code_block in enumerate(code_blocks):
                    test_programs = self._process_test(refer, code_block)

                    # Submit each test program variant for execution
                    for prog_idx, program in enumerate(test_programs):
                        future = executor.submit(
                            execution,
                            program,
                            (
                                i,
                                code_idx,
                                prog_idx,
                            ),  # Pass indices for tracking
                            3,
                        )
                        futures.append(future)

        from tqdm import tqdm

        # Track which examples passed
        passed_examples = set()
        all_results = {}

        for future in tqdm(as_completed(futures), total=len(futures)):
            (example_idx, code_idx, prog_idx), ret, program = future.result()

            # Store result
            if example_idx not in all_results:
                all_results[example_idx] = []

            all_results[example_idx].append({
                'code_idx': code_idx,
                'prog_idx': prog_idx,
                'result': ret,
                'is_correct': ret == 'pass',
                'program': program,
            })

            # If this example passed with any code block or test variant
            if ret == 'pass':
                passed_examples.add(example_idx)

        # Process final results
        for example_idx, results in all_results.items():
            # Did any variant pass?
            example_passed = example_idx in passed_examples

            # Get the first passing result if any, otherwise get the first result
            result_to_use = next((r for r in results if r['is_correct']),
                                 results[0])

            # Update counters
            if example_passed:
                result['pass'] += 1
            else:
                result[result_to_use['result']] += 1

            # Store details
            details[str(example_idx)] = {
                'result':
                ('pass' if example_passed else result_to_use['result']),
                'is_correct': example_passed,
                'num_attempts': len(results),
                'code_blocks_tried': len(set(r['code_idx'] for r in results)),
                'program': result_to_use['program'],
            }

        result['score'] = result['pass'] / len(predictions) * 100
        result['details'] = details
        return result

    def _process_answer(self, text):
        try:
            # for chatGLM related text
            eval_text = eval(text)
        except Exception:
            pass
        else:
            if isinstance(eval_text, str):
                text = eval_text

        code_blocks = []
        # breakpoint()
        # extract all code blocks with ```python or ``` markers
        if '```' in text:
            # Try to find ```python blocks first
            python_blocks = re.findall(r'```python\s*(.*?)```', text,
                                       re.DOTALL)

            # If no ```python blocks, look for generic ``` blocks
            if not python_blocks:
                blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
                if not blocks:
                    # Fall back: split by ``` and take the content between markers
                    parts = text.split('```')
                    if len(parts) > 1:
                        code_blocks.append(parts[1])
                else:
                    for block in blocks:
                        # Skip language identifier if present
                        if not block.startswith('\n') and '\n' in block:
                            block = block[block.find('\n') + 1:]
                        code_blocks.append(block.strip())
            else:
                code_blocks.extend([block.strip() for block in python_blocks])

        # If no code blocks found, use the entire text
        if not code_blocks:
            code_blocks = [text]

        # Process each code block
        processed_blocks = []
        for code in code_blocks:
            # Clean up the code block
            code = code.strip()
            # Remove [BEGIN]/[DONE] markers
            match = re.search(r"('\s*|)(\[DONE\]|DONE)", code)
            if match:
                code = code[:match.start()]
            match = re.search(r"(\[BEGIN\]|BEGIN)('\s*|)", code)
            if match:
                code = code[match.end():]
            code = code.strip()
            if code.startswith("'"):
                code = code[1:]
            if code.endswith("'"):
                code = code[:-1]
            code = code.replace('\\', '')

            processed_blocks.append(code)

        return processed_blocks

    def _process_test(self, test_case, code):
        """Process test with both direct function call and Solution class.

        Args:
            test_case (str): Test case code
            code (str): User submitted code
        """

        # Add wrapper to support Solution class if it exists in the code
        if 'class Solution' in code:
            # Extract the function name from assert statements
            # Looking for patterns like: assert func_name(args)
            func_calls = re.findall(r'assert\s+(\w+)\(', test_case)
            if func_calls:
                # Get unique function names from the test case
                func_names = set(func_calls)

                modified_test = test_case
                for func_name in func_names:
                    # Replace all occurrences of function calls with Solution().func_name
                    modified_test = re.sub(
                        r'(\bassert\s+)' + func_name + r'(\()',
                        r'\1Solution().' + func_name + r'\2',
                        modified_test,
                    )

                # Use the modified test
                test_case = modified_test

        formatted = code + '\n'
        formatted += test_case
        # breakpoint()
        return formatted


def execution(programs, task_ids, timeout):
    """Execution function for running generation code.

    Args:
        programs(str): Python code to be executed.
        task_ids(tuple): Tuple containing (example_idx, code_block_idx, program_variant_idx).
        timeout(int): Time limit for execution.

    Returns:
        tuple: (task_ids, result_status, program_code)
    """

    def _execution(programs, timeout):
        try:
            # Add exec globals to prevent the exec to raise
            # unnecessary NameError for correct answer
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(programs, exec_globals)
            key.append('pass')
        except TimeOutException:
            key.append('timeout')
        except AssertionError:
            key.append('wrong_answer')
        except BaseException as e:
            print(e)
            key.append('failed')

    manager = multiprocessing.Manager()
    key = manager.list()
    # `signal` cannot be used in child thread, therefore, we
    # need to create a process in the thread.
    p = multiprocessing.Process(target=_execution,
                                args=(programs, timeout - 1))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        # key might not have value if killed
        return task_ids, 'timeout', programs
    return task_ids, key[0], programs


class LCPassKEvaluator(LCEvaluator):
    """Better use for pass k evaluation.

    Args:
        k(Tuple[int]): Choices of Pass@k. Defaults to (1, 10, 100)
    """

    def __init__(self, k=(1, 10, 100)) -> None:
        if not isinstance(k, Sequence):
            k = (k, )
        self.k = k

    @staticmethod
    def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int,
    ) -> np.ndarray:
        """Estimates pass@k of each problem and returns them in an array."""

        def estimator(n: int, c: int, k: int) -> float:
            """
            Calculates 1 - comb(n - c, k) / comb(n, k).
            """
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        if isinstance(num_samples, int):
            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            assert len(num_samples) == len(num_correct)
            num_samples_it = iter(num_samples)

        return np.array([
            estimator(int(n), int(c), k)
            for n, c in zip(num_samples_it, num_correct)
        ])

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        task_pass = defaultdict(int)
        task_total = defaultdict(int)

        result = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
        details = {}

        with ProcessPoolExecutor() as executor:
            futures = []
            task_info = []  # Store info for each task

            index = 0
            for refer, preds in zip(references, predictions):
                # suits for two case
                # 1. use repeated dataset
                # 2. use `num_return_sequences` to generate multiple responses
                if not isinstance(preds, list):
                    preds = [preds]

                test_case = refer['test_list_2']
                task_id = refer['task_id']

                # create empty task_pass in case all example failed
                if task_id not in task_pass:
                    task_pass[task_id] = 0

                for pred in preds:
                    # Extract all code blocks from the prediction
                    code_blocks = self._process_answer(pred)

                    # Try each code block with various test program formats
                    for code_idx, code_block in enumerate(code_blocks):
                        # Process test with the current code block
                        test_program = self._process_test(
                            test_case, code_block)

                        # Submit this program for execution
                        future = executor.submit(
                            execution,
                            test_program,
                            (
                                index,
                                task_id,
                                code_idx,
                                0,
                            ),  # prog_idx always 0 since we only have one program per code block
                            30,
                        )
                        futures.append(future)
                        task_info.append({
                            'index': index,
                            'task_id': task_id,
                            'code_block': code_block,
                            'program': test_program,
                        })

                    index += 1

            # Track which tasks have passed with any code block
            passed_tasks = set()
            task_results = defaultdict(list)

            from tqdm import tqdm

            for future in tqdm(as_completed(futures), total=len(futures)):
                (index, task_id, code_idx,
                 prog_idx), ret, program = future.result()

                # Store result
                task_results[(index, task_id)].append({
                    'result': ret,
                    'is_correct': ret == 'pass',
                    'program': program
                })

                # If this is a pass, mark the task
                if ret == 'pass':
                    passed_tasks.add((index, task_id))

                # Store detailed result
                details[f'{index}_{code_idx}_{prog_idx}'] = {
                    'program': program,
                    'task_id': task_id,
                    'result': ret,
                    'is_correct': ret == 'pass',
                }

            # Process all tasks
            for (index, task_id), results in task_results.items():
                task_total[task_id] += 1
                # Task passes if any code block passes
                if (index, task_id) in passed_tasks:
                    task_pass[task_id] += 1
                    result['pass'] += 1
                else:
                    # Get the first result to classify the error
                    first_result = results[0]['result']
                    result[first_result] += 1

        result['details'] = details

        def get_number(tasks):
            return np.array([
                task[1] for task in sorted(tasks.items(), key=lambda x: x[0])
            ])

        task_pass = get_number(task_pass)
        task_total = get_number(task_total)
        pass_at_k = {
            f'pass@{k}':
            self.estimate_pass_at_k(task_total, task_pass, k).mean() * 100
            for k in self.k if (task_total >= k).all()
        }
        result.update(pass_at_k)
        return result
