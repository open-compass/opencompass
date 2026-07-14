"""Dataset and evaluators for the local SciReasoner 1.5 test subset.

The loader reads five user-provided test files:
OQMD/JARVIS-DFT material regression, GO biological-process prediction,
TM-score regression, and DUD-E-like molecule-pair classification.
"""

import json
import math
import os
import random
import re

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

MATERIAL_FILES = {
    'oqmd': 'oqmd_test.json',
    'jarvis_dft': 'jarvis_dft_test.json',
}

TASK_FILES = {
    'go_bp': 'go_test_bp.json',
    'tmscore': 'tmscore_test.json',
    'dude_count': 'dude_count.jsonl',
}


def _read_json_or_jsonl(data_path):
    if data_path.endswith('.jsonl'):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    for item in data:
        yield item


def _reference_dict(reference):
    if isinstance(reference, dict):
        return reference
    if isinstance(reference, str):
        try:
            parsed = json.loads(reference)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {'value': reference}
    return {'value': reference}


def _strip_reasoning(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'<think>.*?</think>',
                  '',
                  text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r'^```(?:json)?\s*|\s*```$',
        '',
        text.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return text.strip()


def extract_scireasoner15_float(text):
    text = _strip_reasoning(text)
    if not text:
        return None
    match = re.search(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?', text)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _extract_property_value(text, property_name):
    text = _strip_reasoning(text)
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and property_name in parsed:
            return float(parsed[property_name])
    except Exception:
        pass

    pattern = (
        r'\{[^{}]*["\']?' + re.escape(property_name) +
        r'["\']?\s*:\s*["\']?([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    # Some models answer with a bare number on regression tasks.
    return extract_scireasoner15_float(text)


def _extract_binary_score(text):
    text = _strip_reasoning(text).lower()
    if not text:
        return None, None, True

    for pattern in ((r'(?:label|answer|prediction|class)\s*[:=]\s*'
                     r'([01])(?=$|\s|[,;}\]])'), ):
        match = re.search(pattern, text)
        if match:
            score = float(match.group(1))
            return int(score >= 0.5), score, False

    number = extract_scireasoner15_float(text)
    if number is not None and 0.0 <= number <= 1.0:
        return int(number >= 0.5), number, False

    positive_words = ('similar', 'active', 'positive', 'yes', 'true')
    negative_words = ('dissimilar', 'inactive', 'negative', 'no', 'false')
    if any(word in text for word in negative_words):
        return 0, 0.0, False
    if any(word in text for word in positive_words):
        return 1, 1.0, False
    return None, None, True


def _split_labels(text):
    if not isinstance(text, str):
        return set()
    text = _strip_reasoning(text)
    if not text:
        return set()
    if ';' in text or '；' in text:
        parts = re.split(r'[;；]\s*', text)
    else:
        parts = re.split(r'[\n,，]\s*', text)
    return {part.strip().lower() for part in parts if part.strip()}


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _rankdata(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _pearson(x_values, y_values):
    if len(x_values) < 2:
        return None
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    numerator = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_den = math.sqrt(sum((x - x_mean)**2 for x in x_values))
    y_den = math.sqrt(sum((y - y_mean)**2 for y in y_values))
    if x_den == 0.0 or y_den == 0.0:
        return None
    return numerator / (x_den * y_den)


def _spearman(x_values, y_values):
    if len(x_values) < 2:
        return None
    return _pearson(_rankdata(x_values), _rankdata(y_values))


def _roc_auc(labels, scores):
    pairs = [(float(score), int(label))
             for label, score in zip(labels, scores) if score is not None]
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None

    sorted_pairs = sorted(pairs, key=lambda item: item[0])
    rank_sum = 0.0
    i = 0
    while i < len(sorted_pairs):
        j = i + 1
        while j < len(
                sorted_pairs) and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        positives_in_tie = sum(label for _, label in sorted_pairs[i:j])
        rank_sum += positives_in_tie * avg_rank
        i = j
    return (rank_sum - positives *
            (positives + 1) / 2.0) / (positives * negatives)


def _regression_scores(y_true, y_pred, total):
    if not y_true:
        return {
            'total': total,
            'valid': 0,
            'valid_rate': 0.0,
            'MAE': None,
            'RMSE': None,
            'MAD': None,
            'MAD/MAE': None,
            'Pearson': None,
            'Spearman': None,
        }
    errors = [pred - true for true, pred in zip(y_true, y_pred)]
    abs_errors = [abs(error) for error in errors]
    mae = _mean(abs_errors)
    rmse = math.sqrt(_mean([error * error for error in errors]))
    true_mean = _mean(y_true)
    mad = _mean([abs(true - true_mean) for true in y_true])
    return {
        'total': total,
        'valid': len(y_true),
        'valid_rate': len(y_true) / total * 100 if total else 0.0,
        'MAE': mae,
        'RMSE': rmse,
        'MAD': mad,
        'MAD/MAE': mad / mae if mae else None,
        'Pearson': _pearson(y_true, y_pred),
        'Spearman': _spearman(y_true, y_pred),
    }


def _select_items(items, mini_set=False, sample_size=None, seed=1024):
    if sample_size is None and not mini_set:
        return items
    limit = sample_size if sample_size is not None else 150
    limit = min(len(items), int(limit))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(items)), limit))
    return [items[idx] for idx in indices]


@LOAD_DATASET.register_module()
class SciReasoner15Dataset(BaseDataset):

    @staticmethod
    def load(path,
             name,
             task_type,
             source=None,
             property_name=None,
             mini_set=False,
             sample_size=None,
             seed=1024,
             **kwargs):
        data_dir = get_data_path(path)
        if task_type == 'material':
            data_file = MATERIAL_FILES[source]
        else:
            data_file = TASK_FILES[name]
        data_path = os.path.join(data_dir, data_file)

        rows = []
        for row_id, item in enumerate(_read_json_or_jsonl(data_path)):
            if task_type == 'material':
                gold = _extract_property_value(item.get('output', ''),
                                               property_name)
                if gold is None:
                    continue
                rows.append({
                    'id': f'{name}-{len(rows)}',
                    'prompt': item['input'],
                    'answer': {
                        'value': gold,
                        'property': property_name,
                        'source': source,
                        'task_type': task_type,
                    },
                    'subset': name,
                })
            elif task_type == 'go_bp':
                rows.append({
                    'id': item.get('name') or f'{name}-{row_id}',
                    'prompt': item['input'],
                    'answer': {
                        'value': item.get('output', ''),
                        'task_type': task_type,
                        'name': item.get('name'),
                        'chain': item.get('chain'),
                    },
                    'subset': name,
                })
            elif task_type == 'tmscore':
                meta = item.get('meta', {})
                rows.append({
                    'id':
                    f'{meta.get("sid1", row_id)}-{meta.get("sid2", row_id)}',
                    'prompt': item['input'],
                    'answer': {
                        'value': float(item.get('output')),
                        'task_type': task_type,
                        'meta': meta,
                    },
                    'subset': name,
                })
            elif task_type == 'dude_count':
                metadata = item.get('metadata', {})
                rows.append({
                    'id': f'{metadata.get("target", "target")}-{row_id}',
                    'prompt': item['input'],
                    'answer': {
                        'value': int(item.get('label')),
                        'task_type': task_type,
                        'target': metadata.get('target'),
                        'metadata': metadata,
                    },
                    'subset': name,
                })
            else:
                raise ValueError(
                    f'Unsupported SciReasoner 1.5 task_type: {task_type}')

        rows = _select_items(rows,
                             mini_set=mini_set,
                             sample_size=sample_size,
                             seed=seed)
        return Dataset.from_list(rows)


@ICL_EVALUATORS.register_module()
class SciReasoner15MaterialEvaluator(BaseEvaluator):

    def __init__(self, property_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.property_name = property_name

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        y_true, y_pred = [], []
        invalid_examples = []
        for idx, (prediction,
                  reference) in enumerate(zip(predictions, references)):
            ref = _reference_dict(reference)
            gold = float(ref['value'])
            pred = _extract_property_value(prediction, self.property_name)
            if pred is None or not math.isfinite(pred):
                if len(invalid_examples) < 20:
                    invalid_examples.append({
                        'index': idx,
                        'prediction': prediction,
                        'answer': gold,
                    })
                continue
            y_true.append(gold)
            y_pred.append(pred)
        result = _regression_scores(y_true, y_pred, len(references))
        result['invalid_count'] = len(references) - len(y_true)
        result['invalid_examples'] = invalid_examples
        return result


@ICL_EVALUATORS.register_module()
class SciReasoner15GOEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        precisions, recalls, f1_scores, intersections = [], [], [], []
        pred_counts, gold_counts = [], []
        exact = 0
        empty_predictions = 0
        for prediction, reference in zip(predictions, references):
            ref = _reference_dict(reference)
            pred_labels = _split_labels(prediction)
            gold_labels = _split_labels(ref['value'])
            if not pred_labels:
                empty_predictions += 1
            overlap = len(pred_labels & gold_labels)
            precision = _safe_ratio(overlap, len(pred_labels))
            recall = _safe_ratio(overlap, len(gold_labels))
            f1 = _safe_ratio(2 * precision * recall, precision + recall)
            exact += int(pred_labels == gold_labels)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            intersections.append(overlap)
            pred_counts.append(len(pred_labels))
            gold_counts.append(len(gold_labels))
        total = len(references)
        return {
            'total':
            total,
            'Precision':
            _mean(precisions),
            'Recall':
            _mean(recalls),
            'F1 Score':
            _mean(f1_scores),
            'score':
            _mean(f1_scores) * 100,
            'exact_match':
            exact / total * 100 if total else 0.0,
            'avg_correct_attributes':
            _mean(intersections),
            'avg_pred_labels':
            _mean(pred_counts),
            'avg_gold_labels':
            _mean(gold_counts),
            'empty_prediction_rate':
            empty_predictions / total * 100 if total else 0.0,
        }


@ICL_EVALUATORS.register_module()
class SciReasoner15TMScoreEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        y_true, y_pred = [], []
        out_of_range = 0
        invalid_examples = []
        for idx, (prediction,
                  reference) in enumerate(zip(predictions, references)):
            ref = _reference_dict(reference)
            gold = float(ref['value'])
            pred = extract_scireasoner15_float(prediction)
            if pred is None:
                if len(invalid_examples) < 20:
                    invalid_examples.append({
                        'index': idx,
                        'prediction': prediction,
                        'answer': gold,
                    })
                continue
            if pred < 0.0 or pred > 1.0:
                out_of_range += 1
            y_true.append(gold)
            y_pred.append(pred)
        result = _regression_scores(y_true, y_pred, len(references))
        result['out_of_range_count'] = out_of_range
        result['invalid_count'] = len(references) - len(y_true)
        result['invalid_examples'] = invalid_examples
        return result


@ICL_EVALUATORS.register_module()
class SciReasoner15DudeEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        labels, preds, scores = [], [], []
        target_stats = {}
        invalid_examples = []
        invalid_count = 0
        for idx, (prediction,
                  reference) in enumerate(zip(predictions, references)):
            ref = _reference_dict(reference)
            label = int(ref['value'])
            pred, score, invalid = _extract_binary_score(prediction)
            if invalid:
                invalid_count += 1
                if len(invalid_examples) < 20:
                    invalid_examples.append({
                        'index': idx,
                        'prediction': prediction,
                        'answer': label,
                    })
                # Default invalid binary outputs to the majority class, while
                # still counting them as invalid in the metrics.
                pred, score = 0, None
            labels.append(label)
            preds.append(pred)
            scores.append(score)

            target = ref.get('target') or 'unknown'
            stats = target_stats.setdefault(target, {
                'tp': 0,
                'fp': 0,
                'tn': 0,
                'fn': 0,
                'total': 0,
            })
            stats['total'] += 1
            if label == 1 and pred == 1:
                stats['tp'] += 1
            elif label == 0 and pred == 1:
                stats['fp'] += 1
            elif label == 0 and pred == 0:
                stats['tn'] += 1
            else:
                stats['fn'] += 1

        tp = sum(1 for label, pred in zip(labels, preds)
                 if label == 1 and pred == 1)
        fp = sum(1 for label, pred in zip(labels, preds)
                 if label == 0 and pred == 1)
        tn = sum(1 for label, pred in zip(labels, preds)
                 if label == 0 and pred == 0)
        fn = sum(1 for label, pred in zip(labels, preds)
                 if label == 1 and pred == 0)
        total = len(labels)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)

        target_metrics = {}
        macro_f1 = []
        macro_recall = []
        for target, stats in target_stats.items():
            target_precision = _safe_ratio(stats['tp'],
                                           stats['tp'] + stats['fp'])
            target_recall = _safe_ratio(stats['tp'], stats['tp'] + stats['fn'])
            target_f1 = _safe_ratio(2 * target_precision * target_recall,
                                    target_precision + target_recall)
            target_accuracy = _safe_ratio(stats['tp'] + stats['tn'],
                                          stats['total'])
            macro_f1.append(target_f1)
            macro_recall.append(target_recall)
            target_metrics[target] = {
                'accuracy': target_accuracy,
                'precision': target_precision,
                'recall': target_recall,
                'f1': target_f1,
                'total': stats['total'],
            }

        return {
            'total': total,
            'Accuracy': _safe_ratio(tp + tn, total) * 100,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'positive_rate': _safe_ratio(sum(preds), total) * 100,
            'gold_positive_rate': _safe_ratio(sum(labels), total) * 100,
            'AUC': _roc_auc(labels, scores),
            'macro_target_f1': _mean(macro_f1),
            'macro_target_recall': _mean(macro_recall),
            'invalid_count': invalid_count,
            'invalid_examples': invalid_examples,
            'target_metrics': target_metrics,
        }
