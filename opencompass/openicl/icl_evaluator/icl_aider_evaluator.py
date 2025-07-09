# flake8: noqa
import concurrent.futures
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict

import requests
from tqdm import tqdm

from .icl_base_evaluator import BaseEvaluator


class AiderEvaluator(BaseEvaluator):

    def __init__(self,
                 language: str = 'py',
                 ip_address: str = 'localhost') -> None:
        self.language = language
        self.url = f'{ip_address}/run_test'
        self.client = requests.Session()
        super().__init__()

    def _send_single_request(self, test_dir: str,
                             prediction: str) -> Dict[str, Any]:
        request = {
            'data': {
                test_dir: prediction
            },
            'model_name': 'default',
            'edit_format': 'whole',
            'no_unit_tests': False,
            'verbose': True
        }

        try:
            response = self.client.post(
                self.url,
                json=request,
                headers={'Content-Type': 'application/json'})
            return {test_dir: response.json()[test_dir]}
        except Exception as e:
            print(f'Error processing {test_dir}: {str(e)}')
            return {test_dir: {'test_outcomes': [False]}}

    def score(self, predictions, references):
        batch_size = 3
        total_correct = 0
        total_count = 0
        details = []

        for i in tqdm(range(0, len(predictions), batch_size),
                      desc='Evaluating batches'):
            batch_predictions = predictions[i:i + batch_size]
            batch_references = references[i:i + batch_size]

            tasks = []
            for prediction, reference in zip(batch_predictions,
                                             batch_references):
                test_dir = reference['test_dir']
                tasks.append((test_dir, prediction))

            batch_results = {}
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=batch_size) as executor:
                future_to_testdir = {
                    executor.submit(self._send_single_request, test_dir,
                                    prediction): test_dir
                    for test_dir, prediction in tasks
                }

                for future in concurrent.futures.as_completed(
                        future_to_testdir):
                    result = future.result()
                    batch_results.update(result)

            print(f'Batch {i//batch_size + 1} results:',
                  batch_results,
                  flush=True)
            for test_dir, outcome in batch_results.items():
                is_correct = outcome['test_outcomes'][0]
                if is_correct:
                    total_correct += 1
                total_count += 1

                details.append({
                    'test_dir': test_dir,
                    'correct': is_correct,
                })

        result = {
            'accuracy':
            100 * total_correct / total_count if total_count > 0 else 0,
            'details': details
        }
        return result
