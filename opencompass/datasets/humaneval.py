# flake8: noqa: E501
# yapf: disable
import copy
import json
import os.path as osp
import re
import tempfile
from os import environ
from typing import List

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

HUMANEVAL_IMPORT_ERROR = '''\
Please install human_eval use following steps:
git clone git@github.com:open-compass/human-eval.git
cd human-eval && pip install -e .'''

HUMANEVAL_PLUS_IMPORT_ERROR = '''\
Please install evalplus use following steps:
git clone --recurse-submodules git@github.com:open-compass/human-eval.git
cd human-eval
pip install -e .
pip install -e evalplus'''


@LOAD_DATASET.register_module()
class HumanevalDataset(BaseDataset):

    @staticmethod
    def load(path: str, num_repeats: int = 1, local_mode: bool = False):
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
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(path, subset_name='openai_humaneval', split='test')
            dataset_list = []
            for example in dataset:
                dataset_list.extend([example] * num_repeats)
            dataset = Dataset.from_list(dataset_list)
        else:
            dataset = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    dataset.extend(
                        [json.loads(line.strip()) for _ in range(num_repeats)])
            dataset = Dataset.from_list(dataset)
        return dataset


class HumanEvalEvaluator(BaseEvaluator):
    """Evaluator for HumanEval or EvalPlus."""

    def __init__(self, k: List[int] = [1, 10, 100]) -> None:
        try:
            import human_eval
        except ImportError:
            raise ImportError(HUMANEVAL_IMPORT_ERROR)

        self.k = k
        super().__init__()

    def score(self, predictions, references, test_set):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        from human_eval.data import HUMAN_EVAL, write_jsonl
        from human_eval.evaluation import evaluate_functional_correctness

        prompts = [item['prompt'] for item in test_set]
        humaneval_preds = []
        # create json file in human_eval format
        for preds, refer in zip(predictions, references):
            # suits for two case
            # 1. use repeated dataset
            # 2. use `num_return_sequences` to generate multiple responses
            if not isinstance(preds, list):
                preds = [preds]
            for pred in preds:
                humaneval_preds.append({'task_id': refer, 'completion': pred})
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = osp.join(tmp_dir, 'human_eval.json')
            write_jsonl(out_dir, humaneval_preds)
            score = evaluate_functional_correctness(out_dir, self.k, n_workers=4, timeout=3.0, problem_file=HUMAN_EVAL)

            detail_path = osp.join(tmp_dir, 'human_eval.json_results.jsonl')
            details = {}
            with open(detail_path, 'r') as f:
                for index, line in enumerate(f):
                    line = json.loads(line)
                    line['is_correct'] = line['passed']
                    line['prompt'] = prompts[index]
                    details[str(index)] = line

        results = {f'humaneval_{k}': score[k] * 100 for k in score}
        results['details'] = details
        return results


class HumanEvalPlusEvaluator(BaseEvaluator):
    """Evaluator for HumanEval or EvalPlus."""

    def __init__(self, k: List[int] = [1, 10, 100]) -> None:
        try:
            import evalplus
        except ImportError:
            raise ImportError(HUMANEVAL_PLUS_IMPORT_ERROR)

        self.k = k
        super().__init__()

    def score(self, predictions, references, test_set):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        from evalplus.data import write_jsonl
        from evalplus.evaluate import evaluate

        prompts = [item['prompt'] for item in test_set]
        humaneval_preds = []
        for preds, refer, prompt in zip(predictions, references, prompts):
            if not isinstance(preds, list):
                preds = [preds]
            for pred in preds:
                humaneval_preds.append({'task_id': refer, 'solution': prompt + pred})
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = osp.join(tmp_dir, 'human_eval.jsonl')
            write_jsonl(out_dir, humaneval_preds)
            flags = dict(
                dataset='humaneval',
                samples=out_dir,
                base_only=None,
                parallel=None,
                i_just_wanna_run=None,
                test_details=0.2,
                min_time_limit=0.2,
                gt_time_limit_factor=4.0,
                mini=None,
            )
            score = evaluate(flags)
            results_path = osp.join(tmp_dir, 'human_eval_eval_results.json')
            with open(results_path, 'r') as f:
                results = json.load(f)
            details = {}
            for index in range(len(predictions)):
                r = results['eval'][references[index]]

                details[str(index)] = {
                    'prompt': prompts[index],
                    'prediction': predictions[index],
                    'reference': references[index],
                    'base_result': r['base'][0][0],
                    'plus_result': r['plus'][0][0],
                    'is_correct': r['base'][0][0] == 'success' and r['plus'][0][0] == 'success',
                }
                if r['nfiles'] > 1:
                    details[str(index)]['warning'] = 'Multiple files in the solution. Details may be wrong.'
        results = {f'humaneval_plus_{k}': score[k] * 100 for k in score}
        results['details'] = details
        return results


def humaneval_postprocess_v2(text: str) -> str:
    blocks = re.findall(r'```\w*\n(.*?)```', text, re.DOTALL)
    if len(blocks) >= 1:
        text = blocks[0]
    return text
