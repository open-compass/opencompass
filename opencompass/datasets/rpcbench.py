"""RPC-Bench: A Fine-grained Benchmark for Research Paper Comprehension.

https://github.com/zai-org/RPC-Bench
https://huggingface.co/datasets/zai-org/RPC-Bench

Text-only (LLM) track: each question is answered with the full MinerU
markdown of the paper in a single-turn prompt.

- Open questions (8 categories) are rated by LLM judges on Correctness,
  Completeness and Conciseness (0-5, one judge request per dimension) and
  aggregated into F1-like / Informativeness as in the official ``eval.py``.
- Claim_Verification questions (True/False) are judged correct/incorrect by
  an LLM judge with ``GenericLLMEvaluator``.
"""
import json
import os
import os.path as osp
import re
from typing import Dict, List, Optional

import mmengine
import numpy as np
from datasets import Dataset
from mmengine.config import ConfigDict

from opencompass.evaluator import GenericLLMEvaluator
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import (ICL_EVALUATORS, ICL_PROMPT_TEMPLATES,
                                  LOAD_DATASET)
from opencompass.utils import build_dataset_from_cfg, build_model_from_cfg
from opencompass.utils.logging import get_logger

from .base import BaseDataset

logger = get_logger(__name__)

CLAIM_VERIFICATION = 'Claim_Verification'


def _resolve_root(path: str, split: str) -> str:
    """Return a local directory holding ``split_metadata/`` and ``md/``.

    ``path`` is either a local copy of the repo or a HF repo id. For a repo
    id only the files needed by ``split`` are fetched (or reused from
    ``HF_DATASETS_CACHE``) with ``snapshot_download``.
    """
    if osp.isdir(path):
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id=path,
        repo_type='dataset',
        allow_patterns=[f'split_metadata/{split}.json', f'md/{split}/**'],
        cache_dir=os.environ.get('HF_DATASETS_CACHE'))


@LOAD_DATASET.register_module()
class RPCBenchDataset(BaseDataset):

    @staticmethod
    def load(path: str, split: str = 'test', subset: str = 'open'):
        """
        Args:
            path: HF repo id (``zai-org/RPC-Bench``) or local repo copy.
            split: ``test`` or ``dev`` (``train`` uses a different schema).
            subset: ``open`` for the 8 open-ended categories, ``cv`` for
                Claim_Verification.
        """
        assert split in ('test', 'dev'), f'unsupported split: {split}'
        assert subset in ('open', 'cv'), f'unsupported subset: {subset}'
        root = _resolve_root(path, split)

        rows = []
        with open(osp.join(root, 'split_metadata', f'{split}.json'),
                  encoding='utf-8') as f:
            papers = [json.loads(line) for line in f if line.strip()]
        for paper in papers:
            md_path = osp.join(root, 'md', split, paper['id'],
                               f"{paper['id']}.md")
            with open(md_path, encoding='utf-8') as f:
                content = f.read()
            # part_idx is the 1-based index within the paper's qa_pairs,
            # same as the official llm.py
            for part_idx, qa in enumerate(paper['qa_pairs'], start=1):
                is_cv = qa['category'] == CLAIM_VERIFICATION
                if is_cv != (subset == 'cv'):
                    continue
                rows.append({
                    'id': paper['id'],
                    'part_idx': part_idx,
                    'title': paper['title'],
                    'abstract': paper['abstract'],
                    'category': qa['category'],
                    'question': qa['question'],
                    'answer': qa['answer'],
                    # Not named `content`: RawPromptTemplate substitutes
                    # columns one after another, and questions/answers contain
                    # LaTeX like `$E_{content}$` that would be replaced too.
                    'paper_content': content,
                })
        return Dataset.from_list(rows)


def _escape_latex(json_str: str) -> str:
    # same as the official eval.py
    return re.sub(r'(?<!\\)\\(?![bfnrtu\\"\'\\])', r'\\\\', json_str)


def parse_rating(judgement: str, dimension: str) -> Optional[float]:
    """Parse ``{dimension: {"rating": "x.xx", ...}}`` from a judge reply.

    Follows the official parsing (```json block + latex escaping +
    ``json.loads(strict=False)``) and falls back to a regex on the rating
    field. Returns None if no rating can be found.
    """
    if not judgement:
        return None
    if '</think>' in judgement:
        judgement = judgement.split('</think>')[-1]
    text = judgement
    match = re.search(r'```json(.*)```', text, re.DOTALL)
    if match:
        text = _escape_latex(match.group(1))
    rating = None
    try:
        content = json.loads(text.strip(), strict=False)
        rating = content.get(dimension, {}).get('rating')
    except (json.JSONDecodeError, AttributeError):
        pass
    if rating is None:
        match = re.search(
            rf'"?{dimension}"?\s*:\s*\{{.*?"rating"\s*:\s*"?([0-9.]+)',
            judgement, re.DOTALL | re.IGNORECASE) or re.search(
                r'"rating"\s*:\s*"?([0-9.]+)', judgement)
        rating = match.group(1) if match else None
    try:
        return min(max(float(rating), 0.0), 5.0)
    except (TypeError, ValueError):
        return None


def _f1_like(correctness: float, completeness: float) -> float:
    if correctness + completeness == 0:
        return 0.0
    return 2 * correctness * completeness / (correctness + completeness)


def _summarize(ratings: Dict[str, List[float]]) -> Dict[str, float]:
    """Official aggregation on a list of 0-5 ratings per dimension.

    F1-like is the harmonic mean of the *averaged* Correctness and
    Completeness; Informativeness = F1-like * Conciseness / 5. All scores
    are scaled to 0-100.
    """
    mean = {dim: float(np.mean(v)) for dim, v in ratings.items()}
    f1 = _f1_like(mean['Correctness'], mean['Completeness'])
    return {
        'f1_like': f1 * 20,
        'informativeness': mean['Conciseness'] * f1 * 4,
        'correctness': mean['Correctness'] * 20,
        'completeness': mean['Completeness'] * 20,
        'conciseness': mean['Conciseness'] * 20,
    }


@ICL_EVALUATORS.register_module()
class RPCBenchLLMEvaluator(GenericLLMEvaluator):
    """Multi-dimension LLM judge for RPC-Bench open questions.

    Every (judge, dimension) pair is a separate judge request, as in the
    official ``eval.py``. Missing / unparsable ratings count as 0, as in the
    official ``get_LLM_score``.

    Args:
        prompt_templates (dict): ``{dimension: prompt_template}`` with
            dimensions ``Correctness``, ``Completeness``, ``Conciseness``.
        judge_cfg (ConfigDict): a single judge model config, or
            ``dict(judgers=[cfg_a, cfg_b])`` to average several judges like
            the official GPT-5 + Gemini setup. Empty falls back to the
            ``OC_JUDGE_*`` environment variables.
        dataset_cfg (ConfigDict): config of the dataset being judged.
    """

    DIMENSIONS = ('Correctness', 'Completeness', 'Conciseness')

    def __init__(
        self,
        prompt_templates: Dict[str, ConfigDict],
        judge_cfg: ConfigDict,
        dataset_cfg: Optional[ConfigDict] = None,
        pred_postprocessor: Optional[ConfigDict] = None,
    ) -> None:
        BaseEvaluator.__init__(self, pred_postprocessor=pred_postprocessor)
        assert set(prompt_templates) == set(self.DIMENSIONS), \
            f'prompt_templates must cover {self.DIMENSIONS}'
        assert dataset_cfg is not None, 'dataset_cfg is None'
        if not judge_cfg:
            self.judge_cfgs = [self.default_judge_cfg]
        elif 'judgers' in judge_cfg:
            self.judge_cfgs = list(judge_cfg['judgers'])
        else:
            self.judge_cfgs = [judge_cfg]
        self.judge_cfg = self.judge_cfgs[0]
        self.prompt_templates = {
            dim: ICL_PROMPT_TEMPLATES.build(tpl)
            for dim, tpl in prompt_templates.items()
        }
        self.dataset_cfg = dataset_cfg
        self.pred_postprocessor = pred_postprocessor

    @staticmethod
    def _judge_name(judge_cfg: ConfigDict) -> str:
        name = judge_cfg.get('abbr') or str(judge_cfg.get('path',
                                                          'judge')).rstrip('/')
        return re.sub(r'[^\w.-]', '_', name.split('/')[-1])

    def _run_judge(self, judge_cfg: ConfigDict, tag: str,
                   retriever: ZeroRetriever) -> Dict[str, Dict]:
        """Run (or reuse) the judge requests of one judge for every
        dimension and return ``{dimension: judge_output}``."""
        outputs, model = {}, None
        for dim in self.DIMENSIONS:
            output_path = (f'{self._out_dir}_replica{self.dataset_replica_idx}'
                           f'{tag}_{dim.lower()}.json')
            if not osp.exists(output_path):
                logger.info(f'LLM judge details will be saved at: '
                            f'{output_path}')
                if model is None:
                    model = build_model_from_cfg(model_cfg=judge_cfg)
                out_dir, out_name = osp.split(output_path)
                inferencer = GenInferencer(
                    model,
                    max_out_len=judge_cfg.get('max_out_len', None),
                    batch_size=judge_cfg.get('batch_size', None),
                    output_json_filepath=out_dir,
                    output_json_filename=out_name,
                )
                inferencer.inference(
                    retriever=retriever,
                    prompt_template=self.prompt_templates[dim])
            outputs[dim] = mmengine.load(output_path)
        return outputs

    def score(self,
              predictions: List,
              references: Optional[List] = None,
              test_set: Optional[Dataset] = None) -> Dict:
        assert len(predictions) == len(
            references), 'predictions and references must have the same length'

        dataset = build_dataset_from_cfg(self.dataset_cfg)
        dataset.reader.dataset['test'] = dataset.test.add_column(
            'prediction', predictions)
        dataset.reader.input_columns.append('prediction')
        retriever = ZeroRetriever(dataset)
        test = dataset.test

        names = [self._judge_name(cfg) for cfg in self.judge_cfgs]
        assert len(set(names)) == len(names), f'duplicate judge names {names}'
        multi_judge = len(self.judge_cfgs) > 1

        # ratings[judge][dim] -> list over samples
        ratings, details = {}, [{
            'id': row['id'],
            'part_idx': row['part_idx'],
            'category': row['category'],
            'question': row['question'],
            'reference': row['answer'],
            'prediction': pred,
            'ratings': {},
            'judge_responses': {},
        } for row, pred in zip(test, predictions)]
        parse_failures = 0
        for name, judge_cfg in zip(names, self.judge_cfgs):
            tag = f'_{name}' if multi_judge else ''
            outputs = self._run_judge(judge_cfg, tag, retriever)
            ratings[name] = {dim: [] for dim in self.DIMENSIONS}
            for i, detail in enumerate(details):
                detail['ratings'][name] = {}
                detail['judge_responses'][name] = {}
                for dim in self.DIMENSIONS:
                    response = outputs[dim][str(i)]['prediction']
                    rating = parse_rating(response, dim)
                    if rating is None:
                        parse_failures += 1
                        rating = 0.0
                    ratings[name][dim].append(rating)
                    detail['ratings'][name][dim] = rating
                    detail['judge_responses'][name][dim] = response

        def summarize(indices: List[int]) -> Dict[str, float]:
            # average the per-judge means, as the official
            # evaluate_two_files does for the GPT and Gemini judges
            per_judge = {
                name: {
                    dim: float(np.mean([r[dim][i] for i in indices]))
                    for dim in self.DIMENSIONS
                }
                for name, r in ratings.items()
            }
            mean = {
                dim: [float(np.mean([v[dim] for v in per_judge.values()]))]
                for dim in self.DIMENSIONS
            }
            return _summarize(mean)

        all_indices = list(range(len(details)))
        results = summarize(all_indices)
        categories = sorted(set(test['category']))
        for cat in categories:
            indices = [i for i, c in enumerate(test['category']) if c == cat]
            for k, v in summarize(indices).items():
                results[f'{cat}/{k}'] = v
        if multi_judge:
            for name, r in ratings.items():
                for k, v in _summarize(r).items():
                    results[f'{name}/{k}'] = v
        results['judge_parse_failures'] = parse_failures
        results['details'] = details
        return results
