"""RebuttalBench: LLM-written author rebuttals scored by an LLM judge.

Paper: Dancing in Chains: Strategic Persuasion in Academic Rebuttal via Theory
of Mind (ICLR 2026), https://arxiv.org/abs/2601.15715
Data:  https://huggingface.co/datasets/RebuttalAgent/RebuttalBench
Code:  https://github.com/Zhitao-He/RebuttalAgent (``eval/``)

Prompts, judge input, judge-output parsing and aggregation follow the official
scripts:

- inference: ``SYS5`` + ``R_input`` of ``ag_eval.py``, the protocol of every
  baseline model in the paper;
- judging: ``SYS4`` + ``RW_input1``, ``json.loads`` on the first
  ```` ```json ```` block or the whole output, up to 5 attempts per sample;
- scores: ``ag_avg.py``.

The prompts themselves are kept verbatim in the dataset configs under
``configs/datasets/RebuttalBench/``.
"""
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
# Row counts of the released files. The paper and dataset card say "2K" for
# RAR_test, but the shipped file holds 2,963 rows.
SPLIT_SIZES = {'RAR_test': 2963, 'R2_test': 6000}

# ``TARGET_CATEGORIES`` / ``DIM_FIELDS`` of ag_avg.py. Comments of any other
# category are generated and judged but left out of every score.
CATEGORIES = (
    'Experimental Rigor',
    'Methodological Soundness',
    'Novelty & Significance',
    'Presentation & Clarity',
)
DIMENSIONS = ('Attitude', 'Clarity', 'Persuasiveness', 'Constructiveness')


def build_judge_input(row: Dict, response: str) -> str:
    """``RW_input1`` of ag_eval.py."""
    return ('The whole review content is\n' + row['review_content'] + '\n' +
            'The comment content is:\n' + row['comment_text'] + '\n' +
            'The original response fragment is\n' + response + '\n' +
            'The best paper fragment is: ' + row['paper_fragment'] + '\n')


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@LOAD_DATASET.register_module()
class RebuttalBenchDataset(BaseDataset):
    """Loader for the two test splits of ``RebuttalAgent/RebuttalBench``.

    Args:
        path: dataset id (``opencompass/rebuttalbench``, resolved by
            ``get_data_path`` and downloaded from OSS if missing), a directory
            holding ``{split}.json`` or a ``*.json`` file.
        split: ``RAR_test`` (out-of-domain) or ``R2_test`` (in-domain).

    Rows without a retrieved paper fragment (72 in ``RAR_test``) are dropped:
    the official scripts index ``top5_text[0]`` and lose those samples.
    """

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
                f'got {len(rows)}. The released files may have changed.')

        samples = []
        for row in rows:
            if not row.get('top5_text'):
                continue
            comment = row['comment']
            samples.append({
                # RAR_test names the review ``content``, R2_test
                # ``review_content``
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
                        'without a paper fragment, as the official scripts do')
        return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Judge parsing and aggregation
# ---------------------------------------------------------------------------


def parse_judge_output(text: Optional[str]) -> Optional[Dict]:
    """``json.loads(extract1(output))`` of the official scripts.

    Returns None when the output does not parse, which triggers another judge
    attempt. A JSON value that is not an object is treated the same way.
    """
    if not isinstance(text, str):
        return None
    matches = re.findall(r'```json(.*?)```', text, re.DOTALL)
    try:
        obj = json.loads(matches[0] if matches else text)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def dimension_scores(judgement: Optional[Dict]) -> Optional[Dict]:
    """The four rubric scores of one parsed judgement, or None if any is
    missing or not a number (the sample is then ignored, as in ag_avg.py).
    Scores are used as given, without clipping."""
    scores = (judgement or {}).get('score', {})
    if not isinstance(scores, dict):
        return None
    values = {dim: _safe_float(scores.get(dim)) for dim in DIMENSIONS}
    if any(v is None for v in values.values()):
        return None
    return values


def _mean(values: List[float]) -> float:
    return mean(values) if values else float('nan')


def aggregate_scores(categories: List[str],
                     judgements: List[Optional[Dict]]) -> Dict[str, float]:
    """``compute_avgs`` of ag_avg.py, on a 0-10 scale.

    - ``score``: mean over all (sample, dimension) pairs, i.e. the
      ``--overall_mode sample`` total;
    - ``score_category_avg``: the ``--overall_mode category`` total, the mean
      over categories of the per-category mean of the four dimensions;
    - ``score_CPCo``: as ``score`` but without Attitude. Not in the official
      code; it reproduces the ``Avg`` column of the paper's result tables;
    - ``{dimension}`` and ``{category}/{dimension}``: the cells of the
      official table.

    Every key is always present (NaN when empty) so that the summarizer sees
    the same metrics for every model.
    """
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
        cat: {
            dim: _mean(buckets[cat][dim])
            for dim in DIMENSIONS
        }
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
    """Scores each response with the official judge prompt and aggregates the
    rubric as ag_avg.py does.

    Args:
        prompt_template (ConfigDict): judge prompt; the ``judge_input`` column
            holds the official ``RW_input1`` text.
        judge_cfg (ConfigDict): judge model config. Empty falls back to the
            ``OC_JUDGE_*`` environment variables. The paper uses Rebuttal-RM
            (https://huggingface.co/RebuttalAgent/Rebuttal-RM).
        max_attempts (int): judge calls per sample. Outputs that do not parse
            are judged again, 5 times at most in the official scripts.
            Samples still unparsed afterwards are left out of every score.
        pred_postprocessor (ConfigDict): applied to the model output before
            judging.
    """

    def __init__(self,
                 prompt_template: ConfigDict,
                 judge_cfg: ConfigDict,
                 max_attempts: int = 5,
                 pred_postprocessor: Optional[ConfigDict] = None) -> None:
        BaseEvaluator.__init__(self, pred_postprocessor=pred_postprocessor)
        self.prompt_template = ICL_PROMPT_TEMPLATES.build(prompt_template)
        self.judge_cfg = judge_cfg
        # e.g. strip the reasoning of a thinking judge before parsing
        self.judge_pred_postprocessor = (judge_cfg
                                         or {}).get('pred_postprocessor', None)
        self.max_attempts = max_attempts

    def _judge(self, judge_inputs: List[str], name: str) -> Dict:
        """Run (or reuse) one judge pass over ``judge_inputs``."""
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
        for attempt in range(self.max_attempts):
            if not pending:
                break
            name = 'judge' if attempt == 0 else f'judge_retry{attempt}'
            judged = self._judge([judge_inputs[i] for i in pending], name)
            still_pending = []
            for j, i in enumerate(pending):
                outputs[i] = judged[str(j)]['prediction']
                attempts[i] = attempt + 1
                judgements[i] = parse_judge_output(outputs[i])
                if judgements[i] is None:
                    still_pending.append(i)
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
