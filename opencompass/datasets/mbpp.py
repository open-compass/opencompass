import contextlib
import io
import itertools
import json
import multiprocessing
import os.path as osp
import re
import signal
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import environ
from typing import List, Sequence, Union

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MBPPDataset(BaseDataset):

    @staticmethod
    def load(path: str, local_mode: bool = False):
        path = get_data_path(path, local_mode=local_mode)

        def processing_test(example):
            example['test_case'] = example['test_list']
            example['test_list'] = '\n'.join(example['test_list'])
            example['test_list_2'] = example['test_list']
            return example

        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            train = MsDataset.load(path,
                                   subset_name='full',
                                   split='train[:10]').map(processing_test)
            test = MsDataset.load(path,
                                  subset_name='full',
                                  split='train[10:510]').map(processing_test)
        else:
            train = load_dataset('json', data_files=path,
                                 split='train[:10]').map(processing_test)
            test = load_dataset('json', data_files=path,
                                split='train[10:510]').map(processing_test)
        return DatasetDict({'train': train, 'test': test})


class MBPPDatasetV2(BaseDataset):

    @staticmethod
    def load(path: str, num_repeats: int = 1):
        """Load mbpp dataset for pass k mode.

        Note that you can use num_repeats > 1 when your model does not support
        `num_return_sequence` in generation, otherwise use the raw
        mbpp dataset and set `num_return_sequence` in model config to
        generate multiple responses for testing pass@k>1.

        It better to change your dataset abbr correspondingly if you want to
        change num_repeats>1, otherwise the number in
        `.cache/dataset_size.json` might be inconsistent.

        Args:
            num_repeats(int): Number of repetition for this dataset to get
        multiple responses in special cases.
        """

        path = get_data_path(path)

        def processing_test(example):
            example['test_case'] = example['test_list']
            example['test_list'] = '\n'.join(example['test_list'])
            example['test_column'] = dict(test_list_2=example['test_list'],
                                          task_id=example['task_id'])
            return example

        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            train = MsDataset.load(path,
                                   subset_name='full',
                                   split='train[:10]').map(processing_test)
            test = MsDataset.load(path,
                                  subset_name='full',
                                  split='train[10:510]').map(processing_test)
        else:
            train = load_dataset('json', data_files=path,
                                 split='train[:10]').map(processing_test)
            test = load_dataset('json', data_files=path,
                                split='train[10:510]').map(processing_test)
        test = concatenate_datasets([test] * num_repeats)
        return DatasetDict({'train': train, 'test': test})


class SanitizedMBPPDataset(BaseDataset):

    @staticmethod
    def load(path: str, num_repeats: int = 1):
        """Load mbpp dataset for pass k mode.

        Note that you can use num_repeats > 1 when your model does not support
        `num_return_sequence` in generation, otherwise use the raw
        mbpp dataset and set `num_return_sequence` in model config to
        generate multiple responses for testing pass@k>1.

        It better to change your dataset abbr correspondingly if you want to
        change num_repeats>1, otherwise the number in
        `.cache/dataset_size.json` might be inconsistent.

        Args:
            num_repeats(int): Number of repetition for this dataset to get
        multiple responses in special cases.
        """
        path = get_data_path(path)

        def processing_test(example):
            example['text'] = example.pop('prompt')
            # used for prompt
            example['test_list'] = '\n'.join(example['test_list'])
            # used for eval
            example['test_list_2'] = example['test_list']
            example['test_column'] = dict(test_list_2=example['test_list'],
                                          task_id=example['task_id'])
            return example

        # train : test = 7 : 257
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            train = MsDataset.load(path,
                                   subset_name='sanitized',
                                   split='train[:7]').map(processing_test)
            test = MsDataset.load(path,
                                  subset_name='sanitized',
                                  split='train[7:264]').map(processing_test)
        else:
            train = load_dataset('json', data_files=path,
                                 split='train[:7]').map(processing_test)
            test = load_dataset('json', data_files=path,
                                split='train[7:264]').map(processing_test)
        test = concatenate_datasets([test] * num_repeats)
        return DatasetDict({'train': train, 'test': test})


class MBPPPlusDataset(BaseDataset):

    @staticmethod
    def load(path: str, num_repeats: int = 1):
        """Load mbpp dataset for pass k mode. Note that you can use
        num_repeats.

        > 1 when your model does not support `num_return_sequence` in
        generation, otherwise use the raw mbpp dataset and set
        `num_return_sequence` in model config to generate multiple responses
        for testing pass@k>1.

        It better to change your dataset abbr correspondingly if you want to
        change num_repeats>1, otherwise the number in
        `.cache/dataset_size.json` might be inconsistent.

        Args:
            num_repeats(int): Number of repetition for this dataset to get
        multiple responses in special cases.
        """

        path = get_data_path(path)

        def processing_test(example):
            example['test_case'] = example['test_list']
            example['test_list'] = '\n'.join(example['test_list'])
            example['test_list_2'] = example['test_list']
            example['test_column'] = dict(test_list_2=example['test_list'],
                                          task_id=example['task_id'])
            return example

        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line.strip())
                example = processing_test(example)
                dataset.extend([example for _ in range(num_repeats)])
        return Dataset.from_list(dataset)


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
class MBPPEvaluator(BaseEvaluator):
    """Evaluator for MBPP or MBPPPlus."""

    def __init__(self, metric: str = 'MBPP') -> None:
        self.metric = metric
        assert self.metric in ['MBPP', 'MBPPPlus']

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        if self.metric == 'MBPP':
            result = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
            details = {}
            with ProcessPoolExecutor() as executor:
                futures = []
                for i, (refer, pred) in enumerate(zip(references,
                                                      predictions)):
                    pred = self._process_answer(pred)
                    programs = self._process_test(refer, pred)
                    future = executor.submit(execution, programs, i, 10)
                    futures.append(future)
                    details[str(i)] = {}
                    details[str(i)]['origin'] = predictions[i]
                    details[str(i)]['programs'] = programs

                from tqdm import tqdm
                for future in tqdm(as_completed(futures), total=len(futures)):
                    index, ret = future.result()
                    result[ret] += 1
                    details[str(index)]['result'] = ret
                    details[str(index)]['is_correct'] = (ret == 'pass')

            result['score'] = result['pass'] / len(predictions) * 100
            result['details'] = details
            return result
        else:
            try:
                from evalplus.data import write_jsonl
                from evalplus.evaluate import evaluate
                self.write_jsonl = write_jsonl
                self.eval = evaluate
            except ImportError:
                raise ImportError(
                    'Please install evalplus use following steps:\n'
                    'git clone --recurse-submodules git@github.com:open-compass/human-eval.git\n'  # noqa
                    'cd human-eval\n'
                    'pip install -e .\n'
                    'pip install -e evalplus\n')
            mbpp_preds = []
            for preds, refer in zip(predictions, references):
                if not isinstance(preds, list):
                    preds = [preds]
                for pred in preds:
                    pred = self._process_answer(pred)
                    mbpp_preds.append({'task_id': refer, 'solution': pred})
            with tempfile.TemporaryDirectory() as tmp_dir:
                out_dir = osp.join(tmp_dir, 'mbpp_eval.jsonl')
                self.write_jsonl(out_dir, mbpp_preds)
                flags = dict(dataset='mbpp',
                             samples=out_dir,
                             base_only=None,
                             parallel=None,
                             i_just_wanna_run=None,
                             test_details=0.2,
                             min_time_limit=0.2,
                             gt_time_limit_factor=4.0,
                             mini=None)
                score = self.eval(flags)
                return {f'mbpp_plus_{k}': score[k] * 100 for k in score}

    def _process_answer(self, text):
        patterns = [
            r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
            r"BEGIN\s*'(.*)'\s*\[DONE\]",
            r"\[BEGIN\]\s*'(.*)'\s*DONE",
            r"BEGIN\s*'(.*)'\s*DONE",
            r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
            r"BEGIN\s*'(.*)\s*\[DONE\]",
            r"\[BEGIN\]\s*'(.*)\s*DONE",
            r"BEGIN\s*'(.*)\s*DONE",
            r'\[BEGIN\]\s*(.*)\s*\[DONE\]',
            r'BEGIN\s*(.*)\s*\[DONE\]',
            r'\[BEGIN\]\s*(.*)\s*DONE',
            r'BEGIN\s*(.*)\s*DONE',
            r'```python\s*(.*)\s*```',
            r'```\s*(.*)\s*```',
            r'```python\s*(.*)\s*$',
            r'```\s*(.*)\s*$',
            r'(.*)\s*```.*',
            r"\[BEGIN\]\s*'(.*)",
            r'\[BEGIN\](.*)',
            r"'(.*)'\s*\[DONE\]",
        ]
        for p in patterns:
            match = re.search(p, text, re.DOTALL)
            if match:
                text = match.group(1)
                break
        text = text.split('```')[0]
        text = re.split(r"'?\s*\[?DONE\]?", text)[0]
        text = text.replace('\\_', '_')
        text = text.strip()
        return text

    def _process_test(self, test_case, pred):
        formatted = pred + '\n'
        formatted += test_case
        return formatted


@ICL_EVALUATORS.register_module()
class MBPPEvaluator2(MBPPEvaluator):
    """Better use for WizardCoder evaluation."""

    def _process_answer(self, text):
        if '```' in text:
            blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
            if len(blocks) == 0:
                text = text.split('```')[1]  # fall back to default strategy
            else:
                text = blocks[0]  # fetch the first code block
                if not text.startswith(
                        '\n'):  # in case starting with ```python
                    text = text[max(text.find('\n') + 1, 0):]
        else:
            match = re.search(r'Here(.*?)\n', text)
            if match:
                text = re.sub('Here(.*?)\n', '', text, count=1)

        # remove test in generation
        test_list = ['# Test', '#Test', '#test', '# test']
        for s in test_list:
            if s in text:
                text = text[:text.find(s)]

        text = text.strip()
        match = re.search(r"('\s*|)(\[DONE\]|DONE)", text)
        if match:
            text = text[:match.start()]
        match = re.search(r"(\[BEGIN\]|BEGIN)('\s*|)", text)
        if match:
            text = text[match.end():]
        text = text.strip()
        if text.startswith("'"):
            text = text[1:]
        return text


def execution(programs, task_id, timeout):
    """Execution function for running generation code.

    Args:
        programs(str): Python code to be executed.
        task_id(int): Task id of the current example.
        timeout(int): Time limit for execution, avoid unnecessary
            blocking.

    In pass@k scenario, a lot of programs should be executed.
    Some internal error cannot be handled properly, such as
    `RecursionError` might cause system break. It is better to
    separate the execution in thread or multiprocess to better
    control the process.
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
        return task_id, 'timeout'
    return task_id, key[0]


class MBPPPassKEvaluator(MBPPEvaluator):
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
        assert len(predictions) == len(references)

        task_pass = defaultdict(int)
        task_total = defaultdict(int)

        result = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
        with ProcessPoolExecutor() as executor:
            futures = []
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
                    pred = self._process_answer(pred)
                    programs = self._process_test(test_case, pred)
                    future = executor.submit(execution, programs, task_id, 10)
                    futures.append(future)

            from tqdm import tqdm
            for future in tqdm(as_completed(futures), total=len(futures)):
                task_id, key = future.result()
                result[key] += 1
                task_total[task_id] += 1
                if key == 'pass':
                    task_pass[task_id] += 1

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
