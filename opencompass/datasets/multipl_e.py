import json
import os.path as osp

from datasets import Dataset

from opencompass.openicl.icl_evaluator.code_evaluator import CodeEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

# currently supporting languages
_HUMANEVAL_LANGUAGE_ = [
    'adb', 'clj', 'cpp', 'cs', 'd', 'dart', 'elixir', 'go', 'hs', 'java', 'jl',
    'js', 'lua', 'ml', 'php', 'pl', 'py', 'r', 'rb', 'rkt', 'rs', 'scala',
    'sh', 'swift', 'ts'
]
_MBPP_LANGUAGE_ = [
    'adb', 'clj', 'cpp', 'cs', 'd', 'elixir', 'go', 'hs', 'java', 'jl', 'js',
    'lua', 'ml', 'php', 'pl', 'py', 'r', 'rb', 'rkt', 'rs', 'scala', 'sh',
    'swift', 'ts'
]


@LOAD_DATASET.register_module()
class MultiplEDataset(BaseDataset):

    @staticmethod
    def load(path: str,
             language: str,
             num_repeats: int = 1,
             tag: str = 'humaneval',
             local_mode: bool = False):
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
        path = get_data_path(path, local_mode=local_mode)
        assert tag in ['humaneval',
                       'mbpp'], 'tag must be in ["humaneval", "mbpp"]'
        if tag == 'humaneval':
            assert language in _HUMANEVAL_LANGUAGE_, (
                f'language must be in {_HUMANEVAL_LANGUAGE_}')
        else:
            assert language in _MBPP_LANGUAGE_, (
                f'language must be in {_MBPP_LANGUAGE_}')
        file_path = osp.join(path, f'{tag}-{language}.jsonl')
        dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.extend(
                    [json.loads(line.strip()) for _ in range(num_repeats)])
        return Dataset.from_list(dataset)


class MultiplEEvaluator(CodeEvaluator):

    def _stop_at_stop_token(self, decoded_string, stop_tokens):
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

    def _process_completions(self, test_case, completions):
        processed_completions = []
        for comp in completions:
            comp = self._extract_code(comp)
            post_comp = self._remove_prefix(test_case['prompt'], comp)
            post_comp = self._stop_at_stop_token(post_comp,
                                                 test_case['stop_tokens'])
            processed_completions.append(post_comp)
        return processed_completions
