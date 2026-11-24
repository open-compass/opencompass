import copy
import json
import os.path as osp
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

NUM_SAMPLES = 1888
# zero-shot prompt of the paper
SUMSURVEY_PROMPT = ('Please summarize the following contents with no more '
                    'than 256 words: ')

DIMENSIONS = ('coverage', 'precision', 'coherence', 'non_redundancy')
WEIGHTS = {
    'coverage': 0.45,
    'precision': 0.30,
    'coherence': 0.15,
    'non_redundancy': 0.10,
}
JUDGE_KEYS = set(DIMENSIONS) | {
    'missing_key_points', 'unsupported_or_conflicting_claims', 'rationale'
}


def build_judge_input(reference: str, candidate: str) -> str:
    return json.dumps(
        {
            'author_abstract': reference,
            'candidate_abstract': candidate
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False)


@LOAD_DATASET.register_module()
class SumSurveyDataset(BaseDataset):

    @staticmethod
    def load(path: str = 'opencompass/sumsurvey') -> Dataset:
        path = get_data_path(path)
        if osp.isdir(path):
            path = osp.join(path, 'sumsurvey.jsonl')
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                samples.append({
                    'id':
                    row['id'],
                    'messages': [{
                        'role': 'user',
                        'content': SUMSURVEY_PROMPT + row['prompt'],
                    }],
                    'reference':
                    row['reference'],
                })
        if len(samples) != NUM_SAMPLES:
            logger.warning(f'SumSurvey: expected {NUM_SAMPLES} rows, got '
                           f'{len(samples)}.')
        return Dataset.from_list(samples)


def parse_judge_output(text: Optional[str]) -> Optional[Dict]:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not isinstance(value, dict) or set(value) != JUDGE_KEYS:
        return None
    for key in DIMENSIONS:
        if type(value[key]) is not int or not 0 <= value[key] <= 4:
            return None
    for key in ('missing_key_points', 'unsupported_or_conflicting_claims'):
        items = value[key]
        if (not isinstance(items, list) or len(items) > 12
                or any(not isinstance(x, str) or not x.strip()
                       for x in items)):
            return None
    if not isinstance(value['rationale'],
                      str) or not value['rationale'].strip():
        return None
    return value


def weighted_score(judgement: Dict) -> float:
    return sum(judgement[key] * WEIGHTS[key] for key in DIMENSIONS) / 4 * 100


def aggregate_scores(statuses: List[str],
                     judgements: List[Optional[Dict]]) -> Dict[str, float]:
    scored = [
        j for status, j in zip(statuses, judgements) if status == 'succeeded'
    ]
    n = len(statuses)
    nan = float('nan')
    sample_scores = [weighted_score(j) for j in scored]
    results = {
        'score': mean(sample_scores) if scored else nan,
        # failed samples counted as 0, not in v1.1
        'score_failures_as_zero': sum(sample_scores) / n if n else nan,
        'scored_ratio': len(scored) / n if n else nan,
        'num_scored': len(scored),
        'source_failed': statuses.count('source_failed'),
        'judge_failed': statuses.count('judge_failed'),
    }
    for key in DIMENSIONS:
        results[f'dim_{key}'] = mean(j[key] for j in scored) if scored else nan
    return results


@ICL_EVALUATORS.register_module()
class SumSurveyEvaluator(GenericLLMEvaluator):

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
            logger.info(f'SumSurvey judge details will be saved at: '
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
        assert test_set is not None, 'SumSurveyEvaluator needs test_set'
        assert len(predictions) == len(test_set), \
            'predictions and test_set must have the same length'
        predictions = ['' if p is None else str(p) for p in predictions]
        refs = test_set['reference']

        # empty outputs are source failures and never reach the judge
        statuses = [
            'judge_failed' if p.strip() else 'source_failed'
            for p in predictions
        ]
        judgements = [None] * len(predictions)
        outputs = [None] * len(predictions)
        attempts = [0] * len(predictions)
        pending = [i for i, s in enumerate(statuses) if s == 'judge_failed']
        for attempt in range(self.max_attempts):
            if not pending:
                break
            name = 'judge' if attempt == 0 else f'judge_retry{attempt}'
            judged = self._judge(
                [build_judge_input(refs[i], predictions[i]) for i in pending],
                name)
            still_pending = []
            for j, i in enumerate(pending):
                outputs[i] = judged[str(j)]['prediction']
                attempts[i] = attempt + 1
                judgements[i] = parse_judge_output(outputs[i])
                if judgements[i] is None:
                    still_pending.append(i)
                else:
                    statuses[i] = 'succeeded'
            pending = still_pending

        results = aggregate_scores(statuses, judgements)

        details = []
        for i, row in enumerate(test_set):
            judgement = judgements[i] or {}
            details.append({
                'id':
                row['id'],
                'status':
                statuses[i],
                'score':
                weighted_score(judgement) if judgements[i] else None,
                'dimensions': {
                    key: judgement[key]
                    for key in DIMENSIONS
                } if judgements[i] else None,
                'missing_key_points':
                judgement.get('missing_key_points'),
                'unsupported_or_conflicting_claims':
                judgement.get('unsupported_or_conflicting_claims'),
                'rationale':
                judgement.get('rationale'),
                'candidate':
                predictions[i],
                'judge_output':
                outputs[i],
                'judge_attempts':
                attempts[i],
            })
        results['details'] = details
        return results


__all__ = [
    'DIMENSIONS',
    'SUMSURVEY_PROMPT',
    'SumSurveyDataset',
    'SumSurveyEvaluator',
    'WEIGHTS',
    'aggregate_scores',
    'build_judge_input',
    'parse_judge_output',
    'weighted_score',
]
