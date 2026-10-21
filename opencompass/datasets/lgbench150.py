"""OpenCompass support for the 150-question LG-Bench subset."""

import json
import re
from pathlib import Path

import numpy as np

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import get_data_path

DOMAINS = ('medical', 'biology', 'chemistry')
DATASET_PATH = 'opencompass/lgbench150'
QUESTION_FILE = 'questions.jsonl'
GRAPH_FILE = 'graph.jsonl'
ANSWER_PATTERN = re.compile(
    r'(?:^|\b)(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*[:：-]?\s*'
    r'\(?([A-D])\)?\b', re.IGNORECASE)


def _load_jsonl(path):
    records = []
    with path.open(encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'Invalid JSONL record at {path}:{line_number}') from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f'Expected a JSON object at {path}:{line_number}')
            records.append(record)
    return records


def _extract_option(text):
    value = str(text).strip().upper()
    matches = ANSWER_PATTERN.findall(value)
    if matches:
        return matches[-1] if len(set(matches)) == 1 else ''
    match = re.fullmatch(r'\(?([A-D])\)?[.。]?', value)
    return match.group(1) if match else ''


def _load_graph(path, categories, threshold):
    sources, targets, weights = [], [], []
    with path.open(encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                edge = json.loads(line)
                source = int(edge['source']) - 1
                target = int(edge['target']) - 1
                weight = float(edge['weight'])
            except (json.JSONDecodeError, KeyError, TypeError,
                    ValueError) as exc:
                raise ValueError(
                    f'Malformed graph edge at {path}:{line_number}') from exc
            if not np.isfinite(weight):
                raise ValueError(
                    f'Non-finite graph weight at {path}:{line_number}')
            if weight > threshold:
                sources.append(source)
                targets.append(target)
                weights.append(weight)

    src = np.asarray(sources, dtype=np.int32)
    dst = np.asarray(targets, dtype=np.int32)
    weight = np.asarray(weights, dtype=np.float64)
    count = len(categories)
    if not len(src) or len(src) != len(dst) or len(src) != len(weight):
        raise ValueError('The filtered graph is empty or malformed')
    if np.any(src == dst):
        raise ValueError('The graph must not contain self-edges')
    if np.any(src < 0) or np.any(dst < 0) or np.any(src >= count) or np.any(
            dst >= count):
        raise ValueError('Graph endpoints must reference question IDs')
    edge_keys = np.minimum(src, dst).astype(
        np.int64) * (count + 1) + np.maximum(src, dst)
    if len(np.unique(edge_keys)) != len(edge_keys):
        raise ValueError('The graph contains duplicate undirected edges')
    return src, dst, weight


def _score_scope(correct, categories, src, dst, weight, domain=None):
    if domain is None:
        scope = np.ones(len(correct), dtype=bool)
        edge_scope = np.ones(len(src), dtype=bool)
    else:
        scope = categories == domain
        edge_scope = scope[src] & scope[dst]
    if not np.any(scope):
        raise ValueError(f'No questions found for domain {domain}')

    scope_src, scope_dst, scope_weight = (src[edge_scope], dst[edge_scope],
                                          weight[edge_scope])
    numerator = (np.bincount(
        scope_src, scope_weight * correct[scope_dst], minlength=len(correct)) +
                 np.bincount(scope_dst,
                             scope_weight * correct[scope_src],
                             minlength=len(correct)))
    denominator = (
        np.bincount(scope_src, scope_weight, minlength=len(correct)) +
        np.bincount(scope_dst, scope_weight, minlength=len(correct)))
    if np.any(denominator[scope] == 0):
        raise ValueError(
            f'Isolated questions found in scope {domain or "overall"}')
    coherence = numerator[scope] / denominator[scope]
    scoped_correct = correct[scope]
    return {
        'Acc': float(100 * scoped_correct.mean()),
        'GCS': float(100 * np.mean(scoped_correct * coherence)),
        'KBS': float(np.var(100 * coherence)),
    }


@ICL_EVALUATORS.register_module()
class LGBench150Evaluator(BaseEvaluator):
    """Calculate Acc, GCS, and KBS on the bundled 150-question subset."""

    def __init__(self,
                 data_dir=DATASET_PATH,
                 questions_path=None,
                 graph_path=None,
                 threshold=0.4):
        super().__init__()
        if questions_path is None or graph_path is None:
            resolved_data_dir = Path(get_data_path(data_dir))
        self.questions_path = (Path(questions_path) if questions_path
                               is not None else resolved_data_dir /
                               QUESTION_FILE)
        self.graph_path = (Path(graph_path) if graph_path is not None else
                           resolved_data_dir / GRAPH_FILE)
        self.threshold = float(threshold)
        if not np.isfinite(self.threshold):
            raise ValueError('Edge threshold must be finite')

    def score(self, predictions, references, test_set):
        questions = _load_jsonl(self.questions_path)
        if len(predictions) != len(questions) or len(references) != len(
                questions) or len(test_set) != len(questions):
            raise ValueError(
                f'Predictions, references, and dataset must each contain '
                f'{len(questions)} rows')
        if [int(row['id'])
                for row in test_set] != [row['id'] for row in questions]:
            raise ValueError('Dataset order does not match graph question IDs')

        gold = [row['answer'].strip().upper() for row in questions]
        if [str(answer).strip().upper() for answer in references] != gold:
            raise ValueError('References do not match question ID order')

        correct = np.asarray([
            int(_extract_option(prediction) == answer)
            for prediction, answer in zip(predictions, gold)
        ],
                             dtype=np.float64)
        categories = np.asarray([row['category'] for row in questions])
        if not set(categories).issubset(DOMAINS):
            raise ValueError('Question data contains an unknown domain')
        src, dst, weight = _load_graph(self.graph_path, categories,
                                       self.threshold)

        metrics = {
            'overall': _score_scope(correct, categories, src, dst, weight)
        }
        for domain in DOMAINS:
            metrics[domain] = _score_scope(correct, categories, src, dst,
                                           weight, domain)
        return {
            f'{scope}/{metric}': round(value, 6)
            for scope, scores in metrics.items()
            for metric, value in scores.items()
        }
