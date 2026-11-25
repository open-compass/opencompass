import copy
import json
import os.path as osp
import re
from statistics import mean
from typing import Dict, List, Optional

import mmengine
from datasets import Dataset
from mmengine.config import ConfigDict

from opencompass.evaluator import GenericLLMEvaluator
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import (ICL_EVALUATORS, ICL_PROMPT_TEMPLATES,
                                  LOAD_DATASET)
from opencompass.utils import build_model_from_cfg, get_data_path
from opencompass.utils.logging import get_logger

from .base import BaseDataset

logger = get_logger()

SPLITS = ('RAR_test', 'R2_test')
SPLIT_SIZES = {'RAR_test': 2963, 'R2_test': 6000}
CATEGORIES = (
    'Experimental Rigor',
    'Methodological Soundness',
    'Novelty & Significance',
    'Presentation & Clarity',
)
DIMENSIONS = ('Attitude', 'Clarity', 'Persuasiveness', 'Constructiveness')


def build_judge_input(row: Dict, response: str) -> str:
    return ('The whole review content is\n' + row['review_content'] + '\n' +
            'The comment content is:\n' + row['comment_text'] + '\n' +
            'The original response fragment is\n' + response + '\n' +
            'The best paper fragment is: ' + row['paper_fragment'] + '\n')


@LOAD_DATASET.register_module()
class RebuttalBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str = 'opencompass/rebuttalbench',
             split: str = 'RAR_test') -> Dataset:
        if split not in SPLITS:
            raise ValueError(f'split must be one of {SPLITS}, got {split!r}')
        path = get_data_path(path)
        filename = path if osp.isfile(path) else osp.join(
            path, f'{split}.json')
        with open(filename, 'r', encoding='utf-8') as f:
            rows = json.load(f)
        if len(rows) != SPLIT_SIZES[split]:
            logger.warning(
                f'RebuttalBench {split}: expected {SPLIT_SIZES[split]} rows, '
                f'got {len(rows)}.')

        samples = []
        for row in rows:
            # the official scripts skip rows without a paper fragment
            if not row.get('top5_text'):
                continue
            comment = row['comment']
            samples.append({
                'review_content':
                row.get('review_content', row.get('content')),
                'comment_text':
                str(comment['comment_text']),
                'paper_fragment':
                row['top5_text'][0],
                'category':
                comment.get('category', ''),
                'sub_category':
                comment.get('sub_category', ''),
                'severity':
                comment.get('severity', ''),
                'paper_id':
                str(row.get('paper_id', '')),
                'comment_id':
                comment.get('comment_id', -1),
            })
        dropped = len(rows) - len(samples)
        if dropped:
            logger.info(f'RebuttalBench {split}: dropped {dropped} rows '
                        'without a paper fragment')
        return Dataset.from_list(samples)


def parse_judge_output(text: Optional[str]) -> Optional[Dict]:
    if not isinstance(text, str):
        return None
    matches = re.findall(r'```json(.*?)```', text, re.DOTALL)
    try:
        obj = json.loads(matches[0] if matches else text)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_score(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if 0 <= score <= 10 else None


def dimension_scores(judgement: Optional[Dict]) -> Optional[Dict]:
    """Return the four scores only when every value is in the rubric range."""
    scores = (judgement or {}).get('score', {})
    if not isinstance(scores, dict):
        return None
    values = {dim: _parse_score(scores.get(dim)) for dim in DIMENSIONS}
    if any(v is None for v in values.values()):
        return None
    return values


def _mean(values: List[float]) -> float:
    return mean(values) if values else float('nan')


def aggregate_scores(categories: List[str],
                     judgements: List[Optional[Dict]]) -> Dict[str, float]:
    buckets = {cat: {dim: [] for dim in DIMENSIONS} for cat in CATEGORIES}
    dim_pool = {dim: [] for dim in DIMENSIONS}
    for category, judgement in zip(categories, judgements):
        if category not in CATEGORIES:
            continue
        values = dimension_scores(judgement)
        if values is None:
            continue
        for dim, value in values.items():
            buckets[category][dim].append(value)
            dim_pool[dim].append(value)

    cat_dim_avg = {
        cat: {dim: _mean(buckets[cat][dim])
              for dim in DIMENSIONS}
        for cat in CATEGORIES
    }
    cat_means = [
        mean(cat_dim_avg[cat].values()) for cat in CATEGORIES
        if buckets[cat][DIMENSIONS[0]]
    ]
    results = {
        'score':
        _mean([v for vs in dim_pool.values() for v in vs]),
        'score_category_avg':
        _mean(cat_means),
        # the Avg column of the paper's tables
        'score_CPCo':
        _mean([
            v for dim in ('Clarity', 'Persuasiveness', 'Constructiveness')
            for v in dim_pool[dim]
        ]),
        'num_scored':
        len(dim_pool[DIMENSIONS[0]]),
    }
    for dim in DIMENSIONS:
        results[dim] = _mean(dim_pool[dim])
    for cat in CATEGORIES:
        for dim in DIMENSIONS:
            results[f'{cat}/{dim}'] = cat_dim_avg[cat][dim]
    return results


@ICL_EVALUATORS.register_module()
class RebuttalBenchEvaluator(GenericLLMEvaluator):

    def __init__(self,
                 prompt_template: ConfigDict,
                 judge_cfg: ConfigDict,
                 max_attempts: int = 5,
                 pred_postprocessor: Optional[ConfigDict] = None) -> None:
        BaseEvaluator.__init__(self, pred_postprocessor=pred_postprocessor)
        self.prompt_template = ICL_PROMPT_TEMPLATES.build(prompt_template)
        self.judge_cfg = judge_cfg
        self.judge_pred_postprocessor = (judge_cfg
                                         or {}).get('pred_postprocessor', None)
        self.max_attempts = max_attempts

    def _judge(self, judge_inputs: List[str], name: str) -> Dict:
        output_path = (f'{self._out_dir}_replica{self.dataset_replica_idx}'
                       f'_{name}.json')
        if not osp.exists(output_path):
            logger.info(f'RebuttalBench judge details will be saved at: '
                        f'{output_path}')
            from opencompass.datasets.lmeval import LMEvalDataset
            dataset = LMEvalDataset(reader_cfg=dict(
                input_columns=['judge_input'],
                output_column=None,
                train_split='test'),
                                    judge_input=judge_inputs)
            judge_cfg = copy.deepcopy(self.judge_cfg or self.default_judge_cfg)
            model = build_model_from_cfg(model_cfg=judge_cfg)
            out_dir, out_name = osp.split(output_path)
            inferencer = GenInferencer(
                model,
                max_out_len=judge_cfg.get('max_out_len', None),
                batch_size=judge_cfg.get('batch_size', None),
                output_json_filepath=out_dir,
                output_json_filename=out_name,
            )
            inferencer.inference(retriever=ZeroRetriever(dataset),
                                 prompt_template=self.prompt_template)
        return self._judge_postprocess(mmengine.load(output_path))

    def score(self,
              predictions: List,
              references: Optional[List] = None,
              test_set: Optional[Dataset] = None) -> Dict:
        assert test_set is not None, 'RebuttalBenchEvaluator needs test_set'
        assert len(predictions) == len(test_set), \
            'predictions and test_set must have the same length'
        predictions = ['' if p is None else str(p) for p in predictions]
        judge_inputs = [
            build_judge_input(row, pred)
            for row, pred in zip(test_set, predictions)
        ]

        judgements = [None] * len(predictions)
        outputs = [None] * len(predictions)
        attempts = [0] * len(predictions)
        pending = list(range(len(predictions)))
        retry_prefix = 'judge_retry'
        for attempt in range(self.max_attempts):
            if not pending:
                break
            name = 'judge' if attempt == 0 else f'{retry_prefix}{attempt}'
            judged = self._judge([judge_inputs[i] for i in pending], name)
            still_pending = []
            for j, i in enumerate(pending):
                outputs[i] = judged[str(j)]['prediction']
                attempts[i] = attempt + 1
                judgements[i] = parse_judge_output(outputs[i])
                if dimension_scores(judgements[i]) is None:
                    still_pending.append(i)
                    if judgements[i] is not None:
                        # Legacy retry files index only JSON parse failures.
                        # Adding this row changes the positions in later files.
                        retry_prefix = 'judge_score_retry'
            pending = still_pending

        categories = test_set['category']
        results = aggregate_scores(categories, judgements)
        results['judge_parse_failures'] = len(pending)

        details = []
        for i, row in enumerate(test_set):
            scores = dimension_scores(judgements[i])
            details.append({
                'paper_id':
                row['paper_id'],
                'comment_id':
                row['comment_id'],
                'category':
                row['category'],
                'response':
                predictions[i],
                'judge_output':
                outputs[i],
                'judge_attempts':
                attempts[i],
                'scores':
                scores,
                'counted':
                scores is not None and row['category'] in CATEGORIES,
            })
        results['details'] = details
        return results


__all__ = [
    'CATEGORIES',
    'DIMENSIONS',
    'RebuttalBenchDataset',
    'RebuttalBenchEvaluator',
    'aggregate_scores',
    'build_judge_input',
    'dimension_scores',
    'parse_judge_output',
]
