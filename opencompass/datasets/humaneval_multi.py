import gzip
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

# currently supporting 19 languages
_LANGUAGE_NAME_DICT = {
    'cpp': 'CPP',
    'cs': 'C#',
    'd': 'D',
    'go': 'Go',
    'java': 'Java',
    'jl': 'Julia',
    'js': 'JavaScript',
    'lua': 'Lua',
    'php': 'PHP',
    'pl': 'Perl',
    'py': 'Python',
    'r': 'R',
    'rb': 'Ruby',
    'rkt': 'Racket',
    'rs': 'Rust',
    'scala': 'Scala',
    'sh': 'Shell',
    'swift': 'Swift',
    'ts': 'TypeScript',
}


@LOAD_DATASET.register_module()
class HumanevalMultiDataset(BaseDataset):

    @staticmethod
    def load(path, language, version, num_repeats: int = 1, **kwargs):
        """Load humaneval dataset for pass k mode.

        Note that you can use num_repeats > 1 when your model does not support
        `num_return_sequence` in generation, otherwise use the raw
        humaneval dataset and set `num_return_sequence` in model config to
        generate multiple responses for testing pass@k>1.

        It better to change your dataset abbr correspondingly if you want to
        change num_repeats>1, otherwise the number in
        `.cache/dataset_size.json` might be inconsistent.

        Args:
            num_repeats(int): Number of repetition for this dataset to get
        multiple responses in special cases.
        """
        path = get_data_path(path, local_mode=True)
        assert language in _LANGUAGE_NAME_DICT.keys(), (
            f'language must be in {list(_LANGUAGE_NAME_DICT.keys())}')
        assert version in [
            'keep', 'transform', 'reworded', 'remove'
        ], ('version must be in ["keep", "transform", "reworded", "remove"]')
        file_path = osp.join(path, f'humaneval-{language}-{version}.jsonl')
        dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.extend(
                    [json.loads(line.strip()) for _ in range(num_repeats)])
        return Dataset.from_list(dataset)


class HumanevalMultiEvaluator(BaseEvaluator):

    def __init__(self,
                 language,
                 ip_address='localhost',
                 port=5000,
                 retry=2,
                 timeout=600) -> None:
        self.language = language
        self.ip_address = ip_address
        self.port = port
        self.retry = retry
        self.timeout = timeout
        super().__init__()

    def stop_at_stop_token(self, decoded_string, stop_tokens):
        """Produces the prefix of decoded_string that ends at the first
        occurrence of a stop_token.

        WARNING: the decoded_string *must not* include the prompt,
        which may have stop tokens itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def _code_eval_service(self, file_path):
        exec_result = subprocess.run([
            'curl', '-X', 'POST', '-F', f'file=@{file_path}', '-F',
            f'dataset=multipl-e/{self.language}',
            f'{self.ip_address}:{self.port}/evaluate'
        ],
                                     timeout=self.timeout,
                                     capture_output=True)

        if exec_result.returncode == 0 and re.match(
                "\"{.*:.*}\"", exec_result.stdout.decode('utf-8')):
            return True, json.loads(exec_result.stdout.decode('utf-8'))
        else:
            if exec_result.stderr:
                try:
                    err = exec_result.stderr.decode()
                except Exception:
                    err = exec_result.stderr
            else:
                try:
                    err = exec_result.stdout.decode()
                except Exception:
                    err = exec_result.stdout
            return False, err

    def estimator(self, n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    def for_file(self, path):

        try:
            with gzip.open(path, 'rt') as f:
                data = json.load(f)
        except Exception:
            return None

        n = len(data['results'])
        c = len([
            True for r in data['results']
            if r['status'] == 'OK' and r['exit_code'] == 0
        ])
        return {
            'pass@1': self.estimator(n, c, 1),
            'pass@10': self.estimator(n, c, 10),
            'pass@100': self.estimator(n, c, 100),
            'n': n,
            'c': c,
        }

    def score(self, predictions, references, test_set):

        stop_tokens = test_set['stop_tokens'][0]
        print(stop_tokens)

        # convert to original version
        test_set = test_set.to_pandas()
        test_set_origin = test_set.drop_duplicates(subset='name')
        num_repeats = int(len(test_set) / len(test_set_origin))
        print(num_repeats)

        # Create a temporary directory using the tempfile module
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(len(test_set_origin)):
                completions = predictions[i * num_repeats:(i + 1) *
                                          num_repeats]
                processed_completions = []
                for comp in completions:
                    comp = self.stop_at_stop_token(comp, stop_tokens)
                    processed_completions.append(comp)

                result_dict = {
                    'name': test_set_origin.iloc[i]['name'],
                    'language': test_set_origin.iloc[i]['language'],
                    'prompt': test_set_origin.iloc[i]['prompt'],
                    'tests': test_set_origin.iloc[i]['tests'],
                    'completions': processed_completions
                }

                json_str = json.dumps(result_dict)
                json_bytes = json_str.encode('utf-8')

                with gzip.GzipFile(
                        os.path.join(tmpdir, f'{result_dict["name"]}.json.gz'),
                        'w') as f:
                    f.write(json_bytes)

            # create a zip file containing all the generated .json.gz files
            zipname = os.path.join(tmpdir, 'archive')
            shutil.make_archive(zipname, 'zip', tmpdir)
            zipfile_path = f'{zipname}.zip'

            num_retry = 0
            while num_retry < self.retry:
                succeed, output = self._code_eval_service(
                    file_path=zipfile_path)
                if not succeed and '(56) Recv failure' in output:
                    # only retry when connection failed
                    num_retry += 1
                    # wait a min in case the service load is too high
                    time.sleep(60)
                else:
                    break

            if succeed:
                if isinstance(output, str):
                    return json.loads(output)
                elif isinstance(output, dict):
                    return output
