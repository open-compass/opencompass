"""HardMTBench: Stress-Testing Chinese-English Translation on Knowledge-
Intensive Domains.

https://github.com/jasonNLP/HardMTBench
https://arxiv.org/abs/2605.28315 (Findings of EMNLP 2026)

10,000 Chinese-English pairs over 12 domains, packaged as 10,000 zh2en and
10,000 en2zh items. The official repo only releases the data; the metrics
follow the paper (Sec. 3.1) and are computed one after another:

- GEMBA-DA (Kocmi & Federmann, 2023) with an LLM judge (gpt-oss-120b in the
  paper). Prompt, answer parsing and the retry-with-higher-temperature loop
  follow the official implementation
  https://github.com/MicrosoftTranslator/GEMBA (gemba/prompt.py,
  gemba/gpt_api.py). The paper names the metric ``GEMBA-DA``, which is the
  reference-free prompt in the official repo (``GEMBA-DA_ref`` is the
  reference-based one), and contrasts it with the reference-based xCOMET.
- xCOMET-XXL (Guerreiro et al., 2024), ``Unbabel/XCOMET-XXL`` served over
  HTTP by the xCOMET service (official ``unbabel-comet`` ``predict``),
  reported x100.
- Terminology accuracy: per item, the fraction of annotated terminology
  pairs whose target-side term appears in the translation after surface
  normalisation, averaged over items.
"""
import copy
import inspect
import json
import os
import os.path as osp
import re
import time
import unicodedata
from typing import Dict, List, Optional, Sequence

import mmengine
import numpy as np
from datasets import Dataset
from mmengine.config import ConfigDict

from opencompass.evaluator import GenericLLMEvaluator
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import (ICL_EVALUATORS, ICL_PROMPT_TEMPLATES,
                                  LOAD_DATASET, MODELS)
from opencompass.utils import build_model_from_cfg, get_data_path
from opencompass.utils.logging import get_logger

from .base import BaseDataset

logger = get_logger(__name__)

DIRECTIONS = ('zh2en', 'en2zh')
DOMAINS = ('Education', 'Finance', 'Gaming', 'History', 'Legal',
           'Media & Entertainment', 'Medical', 'Military', 'News',
           'Papers & Books', 'Sports', 'Technology')
# Chinese names used by the Chinese translation instruction
LANGUAGE_NAMES_ZH = {'Chinese': '中文', 'English': '英语'}
# Hardness buckets of the paper (Sec. 3.6 / Figure 4), left-closed
HARDNESS_BUCKETS = ((50, 70), (70, 80), (80, 90), (90, 100))
METRICS = ('gemba_da', 'xcomet_xxl', 'term_acc')


@LOAD_DATASET.register_module()
class HardMTBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, direction: str, domain: Optional[str] = None):
        """
        Args:
            path: dataset id (``opencompass/hardmtbench``) or the path of
                ``HardMTBench.jsonl``.
            direction: ``zh2en`` or ``en2zh``.
            domain: keep only one of the 12 domains, e.g. ``Technology``.
        """
        assert direction in DIRECTIONS, f'unsupported direction: {direction}'
        assert domain is None or domain in DOMAINS, \
            f'unsupported domain: {domain}'
        path = get_data_path(path)
        rows = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if not item['id'].endswith(f'_{direction}'):
                    continue
                if domain is not None and item['domain'] != domain:
                    continue
                target_language_zh = LANGUAGE_NAMES_ZH[item['target_language']]
                rows.append({
                    'id': item['id'],
                    'domain': item['domain'],
                    'sub_domain': item['sub_domain'],
                    'hardness': float(item['scores']['hardness']),
                    'terminology': item['terminology'],
                    'source_language': item['source_language'],
                    'target_language': item['target_language'],
                    'target_language_zh': target_language_zh,
                    'source_text': item['source_text'],
                    'reference': item['reference'],
                })
        return Dataset.from_list(rows)


def parse_gemba_da(answer: Optional[str]) -> Optional[int]:
    """Port of ``validate_number`` in the official gemba/prompt.py.

    Returns the score in [0, 100], or None when the answer is invalid (no
    or several numbers, or out of range).
    """
    if answer is None:
        return None
    answer = answer.strip()
    numbers = re.findall(r'\d+', answer)
    score = None
    if len(numbers) == 1:
        score = int(numbers[0])
    elif re.match(r"^\[['\"][0-9]*['\"]\]$", answer) is not None:
        score = int(answer[2:-2])
    elif re.match(r'^[0-9]*/100$', answer) is not None:
        score = int(answer.split('/')[0])
    if score is None or score < 0 or score > 100:
        return None
    return score


def normalize_term_text(text: str) -> str:
    """Light surface normalisation for terminology matching: NFKC (folds
    full-width forms), case folding and whitespace collapsing."""
    text = unicodedata.normalize('NFKC', text).casefold()
    return re.sub(r'\s+', ' ', text).strip()


def term_accuracy(prediction: str, terminology: List[Dict],
                  target_language: str) -> Optional[float]:
    """Fraction of annotated terms whose target-side form is in
    ``prediction``.

    The terminology pairs are shared by both directions of a pair and always
    map a Chinese ``source`` to an English ``target``, so the expected term
    is ``target`` for zh2en and ``source`` for en2zh.
    """
    field = 'target' if target_language == 'English' else 'source'
    terms = [normalize_term_text(t[field]) for t in terminology]
    terms = [t for t in terms if t]
    if not terms:
        return None
    prediction = normalize_term_text(prediction)
    return sum(t in prediction for t in terms) / len(terms)


def _hardness_bucket(hardness: float) -> str:
    for low, high in HARDNESS_BUCKETS:
        if hardness < high:
            return f'{low}-{high}'
    low, high = HARDNESS_BUCKETS[-1]
    return f'{low}-{high}'


def _mean(values: Sequence[Optional[float]], scale: float = 1.0) -> float:
    values = [v for v in values if v is not None]
    if not values:
        return float('nan')
    return float(np.mean(values)) * scale


def _set_temperature(judge_cfg: Dict, temperature: float) -> None:
    """Apply the GEMBA sampling temperature to a judge model config."""
    model_type = judge_cfg['type']
    if isinstance(model_type, str):
        model_type = MODELS.get(model_type)
    if 'temperature' in inspect.signature(model_type.__init__).parameters:
        judge_cfg['temperature'] = temperature
    elif 'generation_kwargs' in judge_cfg:
        judge_cfg['generation_kwargs'] = dict(judge_cfg['generation_kwargs'],
                                              temperature=temperature)
    else:
        logger.warning(f'Cannot set temperature={temperature} for judge '
                       f'{model_type.__name__}, its own setting is used.')


@ICL_EVALUATORS.register_module()
class HardMTBenchEvaluator(GenericLLMEvaluator):
    """Computes GEMBA-DA, xCOMET-XXL and terminology accuracy serially.

    Args:
        prompt_template (ConfigDict): GEMBA judge prompt template. Columns
            ``source_language``, ``target_language``, ``source_text``,
            ``reference`` and ``prediction`` are available.
        judge_cfg (ConfigDict): judge model config. Empty falls back to the
            ``OC_JUDGE_*`` environment variables. Its ``temperature`` is
            overridden by the GEMBA schedule (see ``gemba_max_retries``).
        metrics (Sequence[str]): metrics to compute, in order. Defaults to
            all of ``gemba_da``, ``xcomet_xxl`` and ``term_acc``.
        gemba_max_retries (int): like the official ``GptApi.request``,
            answers that cannot be parsed are requested again with
            temperature 0.1, 0.2, ... up to ``gemba_max_retries / 10``
            (official: 1.0). Samples still invalid afterwards are left out
            of the GEMBA mean, as in the official ``Scores.save``.
        xcomet_api_base (str): base URL of the xCOMET scoring service
            (``public_service/xcomet_service``, serving Unbabel/XCOMET-XXL),
            e.g. ``http://host:8090``. Falls back to the
            ``OC_XCOMET_API_BASE`` environment variable. The model is never
            loaded inside the evaluation task.
        xcomet_request_size (int): samples per request to the service.
        xcomet_timeout (int): timeout in seconds of one request.
        xcomet_max_retries (int): retries of a failed request.
    """

    def __init__(
        self,
        prompt_template: ConfigDict,
        judge_cfg: ConfigDict,
        metrics: Sequence[str] = METRICS,
        gemba_max_retries: int = 10,
        xcomet_api_base: Optional[str] = None,
        xcomet_request_size: int = 256,
        xcomet_timeout: int = 3600,
        xcomet_max_retries: int = 5,
        pred_postprocessor: Optional[ConfigDict] = None,
    ) -> None:
        BaseEvaluator.__init__(self, pred_postprocessor=pred_postprocessor)
        assert set(metrics) <= set(METRICS), f'metrics must be in {METRICS}'
        self.metrics = list(metrics)
        self.prompt_template = ICL_PROMPT_TEMPLATES.build(prompt_template)
        self.judge_cfg = judge_cfg
        # e.g. strip the reasoning of a thinking judge before parsing
        self.judge_pred_postprocessor = (judge_cfg
                                         or {}).get('pred_postprocessor', None)
        self.gemba_max_retries = gemba_max_retries
        self.xcomet_api_base = xcomet_api_base
        self.xcomet_request_size = xcomet_request_size
        self.xcomet_timeout = xcomet_timeout
        self.xcomet_max_retries = xcomet_max_retries

    def _output_path(self, name: str) -> str:
        return f'{self._out_dir}_replica{self.dataset_replica_idx}_{name}.json'

    def _judge(self, rows: Dict[str, List], name: str,
               temperature: float) -> Dict:
        """Run (or reuse) one GEMBA judge pass over ``rows``."""
        output_path = self._output_path(name)
        if not osp.exists(output_path):
            logger.info(f'GEMBA judge details will be saved at: '
                        f'{output_path}')
            from opencompass.datasets.lmeval import LMEvalDataset
            dataset = LMEvalDataset(reader_cfg=dict(input_columns=list(rows),
                                                    output_column=None,
                                                    train_split='test'),
                                    **rows)
            judge_cfg = copy.deepcopy(self.judge_cfg or self.default_judge_cfg)
            _set_temperature(judge_cfg, temperature)
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

    def _gemba_da(self, test_set: Dataset, predictions: List[str]) -> Dict:
        scores = [None] * len(predictions)
        responses = [None] * len(predictions)
        pending = list(range(len(predictions)))
        for attempt in range(self.gemba_max_retries + 1):
            if not pending:
                break
            # same placement of the prediction as the official prompt; it is
            # the last column so its content is never substituted again
            rows = {
                col: [test_set[i][col] for i in pending]
                for col in ('source_language', 'target_language',
                            'source_text', 'reference')
            }
            rows['prediction'] = [predictions[i] for i in pending]
            name = 'gemba' if attempt == 0 else f'gemba_retry{attempt}'
            outputs = self._judge(rows, name, temperature=attempt / 10)
            still_pending = []
            for j, i in enumerate(pending):
                response = outputs[str(j)]['prediction']
                responses[i] = response
                score = parse_gemba_da(response)
                if score is None:
                    still_pending.append(i)
                else:
                    scores[i] = score
            pending = still_pending
        return dict(scores=scores, responses=responses, failures=len(pending))

    def _xcomet_xxl(self, test_set: Dataset,
                    predictions: List[str]) -> List[float]:
        """Score with the xCOMET service (see ``xcomet_api_base``), in
        chunks of ``xcomet_request_size`` samples."""
        output_path = self._output_path('xcomet')
        if osp.exists(output_path):
            return mmengine.load(output_path)
        api_base = self.xcomet_api_base or os.environ.get('OC_XCOMET_API_BASE')
        assert api_base, ('xcomet_xxl needs the xCOMET service: set '
                          '`xcomet_api_base` or OC_XCOMET_API_BASE')
        url = api_base.rstrip('/') + '/score'
        samples = [{
            'src': row['source_text'],
            'mt': pred,
            'ref': row['reference'],
        } for row, pred in zip(test_set, predictions)]
        logger.info(f'Scoring {len(samples)} samples with xCOMET at {url}')
        scores = []
        for start in range(0, len(samples), self.xcomet_request_size):
            chunk = samples[start:start + self.xcomet_request_size]
            scores.extend(self._request_xcomet(url, chunk))
            logger.info(f'xCOMET: {len(scores)}/{len(samples)}')
        mmengine.dump(scores, output_path)
        return scores

    def _request_xcomet(self, url: str, samples: List[Dict]) -> List[float]:
        import requests
        for attempt in range(self.xcomet_max_retries + 1):
            try:
                resp = requests.post(url,
                                     json={'samples': samples},
                                     timeout=self.xcomet_timeout)
                resp.raise_for_status()
                scores = resp.json()['scores']
                assert len(scores) == len(samples), \
                    f'got {len(scores)} scores for {len(samples)} samples'
                return [float(s) for s in scores]
            except Exception as e:  # noqa: BLE001
                if attempt == self.xcomet_max_retries:
                    raise RuntimeError(
                        f'xCOMET request to {url} failed') from e
                wait = min(2**attempt, 60)
                logger.warning(f'xCOMET request failed ({e}), retry in '
                               f'{wait}s')
                time.sleep(wait)

    def score(self,
              predictions: List,
              references: Optional[List] = None,
              test_set: Optional[Dataset] = None) -> Dict:
        assert len(predictions) == len(
            references), 'predictions and references must have the same length'
        predictions = [(p or '').strip() for p in predictions]
        details = [{
            'id': row['id'],
            'domain': row['domain'],
            'hardness': row['hardness'],
            'source_text': row['source_text'],
            'reference': row['reference'],
            'prediction': pred,
        } for row, pred in zip(test_set, predictions)]

        per_sample = {}
        results = {}
        # computed serially, in the configured order
        for metric in self.metrics:
            if metric == 'gemba_da':
                gemba = self._gemba_da(test_set, predictions)
                per_sample[metric] = gemba['scores']
                results['gemba_parse_failures'] = gemba['failures']
                for detail, response in zip(details, gemba['responses']):
                    detail['gemba_response'] = response
            elif metric == 'xcomet_xxl':
                per_sample[metric] = [
                    s * 100 for s in self._xcomet_xxl(test_set, predictions)
                ]
            else:
                per_sample[metric] = [
                    term_accuracy(pred, row['terminology'],
                                  row['target_language'])
                    for row, pred in zip(test_set, predictions)
                ]
            for detail, value in zip(details, per_sample[metric]):
                detail[metric] = value

        def summarize(indices: List[int]) -> Dict[str, float]:
            # term_acc is a fraction per item, the others are already 0-100
            return {
                metric:
                _mean([values[i] for i in indices],
                      scale=100 if metric == 'term_acc' else 1)
                for metric, values in per_sample.items()
            }

        summary = summarize(list(range(len(details))))
        domains = test_set['domain']
        if len(set(domains)) > 1:
            for domain in sorted(set(domains)):
                indices = [i for i, d in enumerate(domains) if d == domain]
                for k, v in summarize(indices).items():
                    summary[f'{domain}/{k}'] = v
        buckets = [_hardness_bucket(h) for h in test_set['hardness']]
        for low, high in HARDNESS_BUCKETS:
            bucket = f'{low}-{high}'
            indices = [i for i, b in enumerate(buckets) if b == bucket]
            if indices:
                for k, v in summarize(indices).items():
                    summary[f'hardness_{bucket}/{k}'] = v
        summary.update(results)
        summary['details'] = details
        return summary
