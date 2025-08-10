# Copyright (c) 2024, BigCodeBench and its contributors.
# Copyright (c) 2023, OpenCompass and its contributors.

import os
import time
from concurrent.futures._base import CancelledError

import httpx
from datasets import Dataset, DatasetDict
from gradio_client import Client, handle_file

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.utils import JSONToolkit  # noqa: F401, F403
from opencompass.utils import (check_url_accessibility, get_data_path,
                               get_logger, setup_proxies)

from ..base import BaseDataset
from .extractor import extract_code_generation


class BigCodeBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str = 'opencompass/bigcodebench',
             local_mode: bool = False,
             release_version: str = 'v0.1.2',
             dataset_version: str = 'full'):
        """
        Args:
            path (str): The path to the dataset.
            local_mode (bool): Whether to use local give path or use
                automatically download.
            release_version (str): The release version of the dataset.
            dataset_version (str): The data version of the dataset.
                only support ['full', 'hard']
        """
        assert dataset_version in ['full', 'hard'], \
            'dataset_version should be one of ["full", "hard"], '
        f'but got {dataset_version}'
        path = get_data_path(path, local_mode=local_mode)
        dataset = DatasetDict()
        # Valid Keys:
        # 'task_id', 'complete_prompt', 'instruct_prompt',
        # 'canonical_solution', 'code_prompt', 'test',
        # 'entry_point', 'doc_struct', 'libs'
        if dataset_version == 'full':
            items = JSONToolkit.read_jsonl(
                os.path.join(path, f'BigCodeBench-{release_version}.jsonl'))
        else:
            items = JSONToolkit.read_jsonl(
                os.path.join(path,
                             f'BigCodeBench-Hard-{release_version}.jsonl'))

        dataset['train'] = Dataset.from_list(items)
        dataset['test'] = Dataset.from_list(items)

        return dataset


class BigCodeBenchEvaluator(BaseEvaluator):
    """Evaluator for BigCodeBench.

    Args:
        num_process_evaluate (int): number of processes to evaluate
        timeout (int): timeout for each evaluation
        release_version (str): release version of BigCodeBench
        eval_type (str): type of evaluation, either 'instruct' or 'completion'
    """

    def __init__(
            self,
            release_version='v0.1.2',
            eval_type='instruct',
            remote_execute_api='https://bigcode-bigcodebench-evaluator.hf.space/',  # noqa
            dataset_version: str = 'full',
            local_mode: bool = False,
            path: str = 'opencompass/bigcodebench',
            pass_k: str = '1,5,10',
            parallel: int = -1,
            min_time_limit: float = 1,
            max_as_limit: int = 30 * 1024,
            max_data_limit: int = 30 * 1024,
            max_stack_limit: int = 10,
            check_gt_only: bool = False,
            no_gt: bool = False):
        super().__init__()
        self.dataset = BigCodeBenchDataset.load(
            release_version=release_version,
            dataset_version=dataset_version,
            local_mode=local_mode,
            path=path)['test']
        self.eval_type = eval_type
        self.remote_execute_api = remote_execute_api

        self.eval_kwargs = dict(subset=dataset_version,
                                pass_k=pass_k,
                                parallel=parallel,
                                min_time_limit=min_time_limit,
                                max_as_limit=max_as_limit,
                                max_data_limit=max_data_limit,
                                max_stack_limit=max_stack_limit,
                                check_gt_only=check_gt_only,
                                no_gt=no_gt)

    def score(self, predictions, references):
        logger = get_logger()
        entrypoints = [item['entry_point'] for item in self.dataset]

        # Append content to the end of the prompt for Completion
        if self.eval_type == 'complete':
            content = [item['complete_prompt'] for item in self.dataset]
            predictions = [
                content[idx] + item for idx, item in enumerate(predictions)
            ]
        elif self.eval_type == 'instruct':
            pass
        else:
            raise ValueError(f'Unknown eval_type: {self.eval_type}')

        # Sanitize predictions for execution
        logger.info('Start to extract code from predictions')
        sanitized_predictions = []
        for prediction, entrypoint in zip(predictions, entrypoints):
            try:
                import signal
                from contextlib import contextmanager

                @contextmanager
                def timeout_handler(seconds):

                    def _handle_timeout(signum, frame):
                        raise TimeoutError(f'Code extraction timed out'
                                           f'after {seconds} seconds')

                    original_handler = signal.signal(signal.SIGALRM,
                                                     _handle_timeout)
                    signal.alarm(seconds)
                    try:
                        yield
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, original_handler)

                with timeout_handler(10):
                    sanitized_prediction = extract_code_generation(
                        prediction, entrypoint=entrypoint)

            except TimeoutError as e:
                logger.warning(
                    f'Code extraction timeout for entrypoint {entrypoint}: '
                    f'{str(e)}')
                sanitized_prediction = ''
            except Exception as e:
                logger.warning(
                    f'Code extraction failed for entrypoint {entrypoint}: '
                    f'{str(e)}')
                sanitized_prediction = ''
            sanitized_predictions.append(sanitized_prediction)

        # Prepare for submission
        submitted_contents = []
        task_ids = [item['task_id'] for item in self.dataset]
        for task_id, sanitized_prediction in zip(task_ids,
                                                 sanitized_predictions):
            submitted_content = {
                'task_id': task_id,
                'solution': sanitized_prediction
            }
            submitted_contents.append(submitted_content)

        submitted_contents_path = os.path.join(
            self._out_dir, 'bigcodebench_submitted_contents.jsonl')
        JSONToolkit.save_jsonl(submitted_contents, submitted_contents_path)
        logger.info(f'Dump submitted contents to {submitted_contents_path}')

        logger.info(
            f'Start to connect to {self.remote_execute_api} for evaluating')
        # Conduct evaluation with Eval Client
        proxies = setup_proxies('BIGCODEBENCH_EVAL_PROXY_URL')

        is_accessible, status_code = check_url_accessibility(
            self.remote_execute_api)
        if not is_accessible:
            logger.error(f'Failed to connect to {self.remote_execute_api} '
                         f'with status code {status_code}')
            return False

        while True:
            try:
                eval_client = Client(self.remote_execute_api,
                                     httpx_kwargs=dict(
                                         proxies=proxies,
                                         timeout=httpx.Timeout(100.0)))
                results, pass_at_k = eval_client.predict(
                    split=self.eval_type,
                    samples=handle_file(submitted_contents_path),
                    api_name='/predict',
                    **self.eval_kwargs)
                break
            except (httpx.ReadTimeout, CancelledError):
                logger.info('Read timeout error. Retrying in 10s...')
                time.sleep(10)

        if 'pass@1' in pass_at_k.keys():
            pass_at_k['pass@1'] *= 100
        dump_results = {'details': self._results_processor(results)}
        dump_results.update(pass_at_k)

        return dump_results

    def _results_processor(self, results):
        details = []
        for key, value in results['eval'].items():
            if value[0]['status'] == 'pass':
                value[0]['correct'] = True
            else:
                value[0]['correct'] = False
            details.append(value[0])
        return details
