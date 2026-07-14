import gzip
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import tabulate
from mmengine import ConfigDict, ProgressBar

from opencompass.utils.abbr import (dataset_abbr_from_cfg,
                                    get_infer_output_path, model_abbr_from_cfg)

PATTERN_SIZE = 32
NUM_SAMPLE_PATTERNS = 5
MIN_PATTERN_COUNT = 20
MIN_PATTERN_INTERVAL_CONCENTRATION = 0.9
GZIP_THRESHOLD = 40.0

_tokenizer_cache: Dict[str, 'TokenizerWrapper'] = {}

_mp_tokenizer = None


def _mp_init_tokenizer(tokenizer_name: str):
    """Initialize tokenizer in child process by name."""
    global _mp_tokenizer
    from transformers import AutoTokenizer
    _mp_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                  trust_remote_code=True)


def _mp_tokenize_batch(texts: List[str]) -> List[List[str]]:
    return [
        _mp_tokenizer.convert_ids_to_tokens(
            _mp_tokenizer(t, add_special_tokens=True).input_ids) for t in texts
    ]


def _mp_count_batch(texts: List[str]) -> List[int]:
    return [
        len(_mp_tokenizer(t, add_special_tokens=True).input_ids) for t in texts
    ]


class TokenizerWrapper:

    def __init__(self,
                 tokenizer,
                 tokenizer_type: str,
                 tokenizer_name: str = ''):
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.tokenizer_name = tokenizer_name
        self.is_fast = (tokenizer_type == 'tiktoken'
                        or getattr(tokenizer, 'is_fast', False))
        self._pool = None
        self._pool_workers = 0

    def _get_pool(self, max_workers: Optional[int] = None):
        workers = max_workers or 8
        if self._pool is None or self._pool_workers != workers:
            self.close()
            from multiprocessing import get_context
            ctx = get_context('spawn')
            self._pool = ctx.Pool(workers,
                                  initializer=_mp_init_tokenizer,
                                  initargs=(self.tokenizer_name, ))
            self._pool_workers = workers
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
            self._pool_workers = 0

    def tokenize(self, text: str) -> List[str]:
        if self.tokenizer_type == 'tiktoken':
            return [
                self.tokenizer.decode([t])
                for t in self.tokenizer.encode(text, disallowed_special=())
            ]
        ids = self.tokenizer(text, add_special_tokens=True).input_ids
        return self.tokenizer.convert_ids_to_tokens(ids)

    def iter_batch_tokenize(
        self,
        texts: List[str],
        batch_size: int = 512,
        max_workers: Optional[int] = None,
    ) -> Iterator[List[List[str]]]:
        """Yield tokenized results batch by batch."""
        if self.tokenizer_type == 'tiktoken':
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = self.tokenizer.encode_batch(batch,
                                                      disallowed_special=())
                yield [[self.tokenizer.decode([t]) for t in ids]
                       for ids in encoded]
        elif self.is_fast:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                batch_encoding = self.tokenizer(batch, add_special_tokens=True)
                yield [
                    self.tokenizer.convert_ids_to_tokens(ids)
                    for ids in batch_encoding['input_ids']
                ]
        else:
            pool = self._get_pool(max_workers)
            workers = self._pool_workers
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                chunk_size = max(1, len(batch) // workers)
                chunks = [
                    batch[i:i + chunk_size]
                    for i in range(0, len(batch), chunk_size)
                ]
                results = pool.map(_mp_tokenize_batch, chunks)
                yield [tok for chunk_result in results for tok in chunk_result]

    def batch_count(self,
                    texts: List[str],
                    batch_size: int = 512,
                    max_workers: Optional[int] = None) -> List[int]:
        results: List[int] = []
        if self.tokenizer_type == 'tiktoken':
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = self.tokenizer.encode_batch(batch,
                                                      disallowed_special=())
                results.extend(len(ids) for ids in encoded)
        elif self.is_fast:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                batch_encoding = self.tokenizer(batch, add_special_tokens=True)
                results.extend(len(ids) for ids in batch_encoding['input_ids'])
        else:
            pool = self._get_pool(max_workers)
            workers = self._pool_workers
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                chunk_size = max(1, len(batch) // workers)
                chunks = [
                    batch[i:i + chunk_size]
                    for i in range(0, len(batch), chunk_size)
                ]
                for chunk_counts in pool.map(_mp_count_batch, chunks):
                    results.extend(chunk_counts)
        return results


def _build_tokenizer_from_model_cfg(model_cfg: ConfigDict) -> TokenizerWrapper:
    model_abbr = model_abbr_from_cfg(model_cfg)
    if model_abbr in _tokenizer_cache:
        return _tokenizer_cache[model_abbr]

    tokenizer_path = model_cfg.get('tokenizer_path', None)
    path = model_cfg.get('path', '')
    name = tokenizer_path or path

    try:
        import tiktoken
        if name in tiktoken.model.MODEL_TO_ENCODING:
            tok = tiktoken.encoding_for_model(name)
            wrapper = TokenizerWrapper(tok, 'tiktoken', name)
            _tokenizer_cache[model_abbr] = wrapper
            return wrapper
    except Exception:
        pass

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        wrapper = TokenizerWrapper(tok, 'hf', name)
        _tokenizer_cache[model_abbr] = wrapper
        return wrapper
    except Exception:
        pass

    try:
        import tiktoken
        tok = tiktoken.encoding_for_model('gpt-4')
        wrapper = TokenizerWrapper(tok, 'tiktoken', 'gpt-4')
        _tokenizer_cache[model_abbr] = wrapper
        return wrapper
    except Exception:
        pass

    raise RuntimeError(f'Cannot initialize tokenizer for model {model_abbr}')


def _prediction_to_text(prediction: Any) -> str:
    if prediction is None:
        return ''
    if isinstance(prediction, str):
        return prediction
    if isinstance(prediction, (int, float, bool)):
        return str(prediction)
    if isinstance(prediction, list):
        return '\n'.join(_prediction_to_text(item) for item in prediction)
    if isinstance(prediction, dict):
        if 'prediction' in prediction:
            return _prediction_to_text(prediction['prediction'])
        return json.dumps(prediction,
                          ensure_ascii=False,
                          sort_keys=True,
                          default=str)
    return str(prediction)


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    values = list(values)
    if not values:
        return 0, 0
    mean = sum(values) / len(values)
    variance = sum((value - mean)**2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _empty_repeat_pattern_stats() -> Dict[str, Any]:
    return {
        'repeat_pattern_ratio': 0,
        'repeat_pattern': '',
        'repeat_pattern_count': 0,
    }


def _repeat_pattern_stats(
    tokens: List[str],
    pattern_size: int = PATTERN_SIZE,
    num_samples: int = NUM_SAMPLE_PATTERNS,
) -> Dict[str, Any]:
    """Detect periodic repetition by sampling windows from the tail.

    Sample windows from the tail of the token sequence. For each window,
    find all occurrence positions, compute the intervals between consecutive
    occurrences, and check if the top-2 most common intervals dominate.

    A pattern is considered periodic if:
    - It occurs at least 3 times.
    - The top-2 interval values account for >= 80% of all intervals.

    The repeat_pattern_ratio is: top2_interval_ratio * count / total_tokens,
    combining periodicity confidence with coverage.
    """
    n = len(tokens)
    if n < pattern_size * 2:
        return _empty_repeat_pattern_stats()

    vocab: Dict[str, int] = {}
    ids = np.empty(n, dtype=np.int32)
    for i, tok in enumerate(tokens):
        if tok not in vocab:
            vocab[tok] = len(vocab)
        ids[i] = vocab[tok]

    windows = np.lib.stride_tricks.sliding_window_view(ids, pattern_size)

    tail_start = max(n // 2, n - pattern_size * (num_samples + 1))
    available = n - tail_start - pattern_size + 1
    if available <= 0:
        return _empty_repeat_pattern_stats()

    step = max(1, available // num_samples)
    best_ratio = 0.0
    best_offset = tail_start
    best_count = 0

    seen_hashes: set = set()
    for i in range(tail_start, tail_start + available, step):
        target = ids[i:i + pattern_size]
        h = hash(target.tobytes())
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        matches = np.flatnonzero(np.all(windows == target, axis=1))
        count = len(matches)
        if count < MIN_PATTERN_COUNT:
            continue

        intervals = np.diff(matches)
        unique, counts = np.unique(intervals, return_counts=True)
        top2_count = int(counts[np.argsort(counts)[-2:]].sum())
        top2_ratio = top2_count / len(intervals)
        if top2_ratio < MIN_PATTERN_INTERVAL_CONCENTRATION:
            continue

        ratio = top2_ratio * count * pattern_size / n
        if ratio > best_ratio:
            best_ratio = ratio
            best_offset = i
            best_count = count

    best_pattern = tokens[best_offset:best_offset + pattern_size]
    return {
        'repeat_pattern_ratio': min(best_ratio, 1.0),
        'repeat_pattern': ' '.join(best_pattern),
        'repeat_pattern_count': best_count,
    }


def _gzip_compression_ratio(text: str) -> float:
    raw = text.encode('utf-8')
    if not raw:
        return 0
    compressed = gzip.compress(raw)
    return len(raw) / len(compressed)


def _iter_prediction_items(path: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict):
        for sample_id, sample in data.items():
            yield str(sample_id), sample
    elif isinstance(data, list):
        for index, sample in enumerate(data):
            yield str(index), sample
    else:
        yield '0', {'prediction': data}


def _collect_predictions_by_benchmark(
    config: ConfigDict,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, str]]]:
    """Read all prediction files, grouped by benchmark.

    Each record has: model, benchmark, sample_id, prediction_path, text,
    and optionally res_length.

    Returns:
        (grouped_records, missing_files)
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    missing_files = []
    prediction_root = Path(config['work_dir']) / 'predictions'

    for model in config['models']:
        model_abbr = model_abbr_from_cfg(model)
        for dataset in config['datasets']:
            benchmark = dataset_abbr_from_cfg(dataset)
            path = Path(
                get_infer_output_path(model, dataset, str(prediction_root)))
            # Support sharded prediction files: abbr_0.json, abbr_1.json, ...
            if path.exists():
                paths = [path]
            else:
                paths = sorted(path.parent.glob(f'{path.stem}_*.json'))
            if not paths:
                missing_files.append({
                    'model': model_abbr,
                    'benchmark': benchmark,
                    'path': str(path),
                })
                continue

            for pred_path in paths:
                for sample_id, sample in _iter_prediction_items(pred_path):
                    res_length: Optional[int] = None
                    if isinstance(sample, dict):
                        prediction = sample.get('prediction', '')
                        raw_res_length = sample.get('res_length', None)
                        if raw_res_length is not None:
                            if isinstance(raw_res_length, list):
                                res_length = (raw_res_length[0]
                                              if raw_res_length else 0)
                            else:
                                res_length = raw_res_length
                    else:
                        prediction = sample
                    text = _prediction_to_text(prediction)

                    grouped[benchmark].append({
                        'model': model_abbr,
                        'benchmark': benchmark,
                        'sample_id': sample_id,
                        'prediction_path': str(pred_path),
                        'text': text,
                        'res_length': res_length,
                    })
    return grouped, missing_files


def _analyze_benchmark(
    records: List[Dict[str, Any]],
    tokenizer_map: Dict[str, TokenizerWrapper],
    show_progress: bool = False,
    batch_size: int = 512,
    think_tag: str | None = None,
) -> Dict[str, Any]:
    """Analyze one benchmark's records. Returns benchmark stats dict.

    Steps:
    1. Batch tokenize all records to get token lengths
       (use res_length if available).
    2. Compute length threshold (mean + std).
    3. Identify candidates above threshold.
    4. Batch tokenize candidates that only had res_length
       (need tokens for pattern).
    5. Compute pattern/gzip for all candidates.
    6. Discard token data, keep only metrics on records.
    """

    # --- Step 1: compute token lengths ---
    # Separate records by whether they have res_length
    needs_tokenize: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        if record['res_length'] is not None and think_tag is None:
            record['length'] = record['res_length']
        else:
            needs_tokenize[record['model']].append(idx)

    # Batch tokenize to get lengths (and cache tokens for later)
    total_tokenize = sum(len(indices) for indices in needs_tokenize.values())
    progress_bar = (ProgressBar(total_tokenize)
                    if show_progress and total_tokenize > 0 else None)

    # _tokens is temporarily cached per record for candidates
    for model_abbr, indices in needs_tokenize.items():
        tokenizer = tokenizer_map[model_abbr]
        if think_tag:
            think_token = tokenizer.tokenize(think_tag)[0]
        else:
            think_token = None
        texts = [records[i]['text'] for i in indices]
        for batch_tokens in tokenizer.iter_batch_tokenize(
                texts, batch_size=batch_size):
            batch_indices = indices[:len(batch_tokens)]
            indices = indices[len(batch_tokens):]
            for idx, tokens in zip(batch_indices, batch_tokens):
                records[idx]['length'] = len(tokens)
                records[idx]['_tokens'] = tokens
                if think_token:
                    records[idx]['reasoning_length'] = 0
                    records[idx]['content_length'] = len(tokens)
                    if think_token in tokens:
                        records[idx]['reasoning_length'] = tokens.index(
                            think_token)
                        records[idx]['content_length'] = len(
                            tokens) - tokens.index(think_token) - 1
                if progress_bar:
                    progress_bar.update()

    # --- Step 2: compute threshold ---
    lengths = sorted(r['length'] for r in records)
    length_mean = sum(lengths) / len(lengths) if lengths else 0
    length_p75 = lengths[int(len(lengths) * 0.75)] if lengths else 0
    length_p90 = lengths[int(len(lengths) * 0.90)] if lengths else 0
    repeat_threshold = length_p75

    # --- Step 3: identify candidates ---
    candidates: List[int] = []
    for idx, record in enumerate(records):
        if record['length'] >= repeat_threshold:
            candidates.append(idx)

    # --- Step 4: tokenize candidates that had res_length (no _tokens yet) ---
    needs_candidate_tokenize: Dict[str, List[int]] = defaultdict(list)
    for idx in candidates:
        if '_tokens' not in records[idx]:
            needs_candidate_tokenize[records[idx]['model']].append(idx)

    total_candidate_tok = sum(
        len(v) for v in needs_candidate_tokenize.values())
    if total_candidate_tok > 0:
        progress_bar = (ProgressBar(total_candidate_tok)
                        if show_progress else None)
        for model_abbr, indices in needs_candidate_tokenize.items():
            tokenizer = tokenizer_map[model_abbr]
            texts = [records[i]['text'] for i in indices]
            for batch_tokens in tokenizer.iter_batch_tokenize(
                    texts, batch_size=batch_size):
                batch_indices = indices[:len(batch_tokens)]
                indices = indices[len(batch_tokens):]
                for idx, tokens in zip(batch_indices, batch_tokens):
                    records[idx]['_tokens'] = tokens
                    if progress_bar:
                        progress_bar.update()

    # --- Step 5: pattern/gzip ---
    # Initialize all records with empty metrics
    for record in records:
        record.update({
            'gzip_compression_ratio': 0,
            'repeat_analysis_skipped': True,
            **_empty_repeat_pattern_stats(),
        })

    pattern_time = 0.0
    gzip_time = 0.0

    if candidates:
        work_items = [(records[idx]['text'], records[idx]['_tokens'])
                      for idx in candidates]

        progress_bar = (ProgressBar(len(candidates))
                        if show_progress else None)

        for idx, (text, tokens) in zip(candidates, work_items):
            half_tokens = tokens[len(tokens) // 2:]
            half_text = text[len(text) // 2:]
            t0 = time.perf_counter()
            pattern_stats = _repeat_pattern_stats(half_tokens)
            t1 = time.perf_counter()
            gzip_ratio = _gzip_compression_ratio(half_text)
            t2 = time.perf_counter()

            pattern_time += t1 - t0
            gzip_time += t2 - t1

            records[idx].update(pattern_stats)
            records[idx]['gzip_compression_ratio'] = gzip_ratio
            records[idx]['repeat_analysis_skipped'] = False
            if progress_bar:
                progress_bar.update()

    # --- Step 6: cleanup, free tokenize memory ---
    for record in records:
        record.pop('_tokens', None)
        record.pop('text', None)
        record.pop('res_length', None)

    # Build benchmark stats
    pattern_mean, pattern_std = _mean_std(r['repeat_pattern_ratio']
                                          for r in records)
    gzip_mean = (sum(r['gzip_compression_ratio']
                     for r in records) / len(records)) if records else 0

    res = {
        'sample_count':
        len(records),
        'repeat_analysis_skipped':
        sum(r['repeat_analysis_skipped'] for r in records),
        'length': {
            'mean': length_mean,
            'p75': length_p75,
            'p90': length_p90,
            'repeat_analysis_threshold': repeat_threshold,
        },
        'repeat_pattern_ratio': {
            'mean': pattern_mean,
            'std': pattern_std,
            'threshold': MIN_PATTERN_COUNT,
        },
        'gzip_compression_ratio': {
            'mean': gzip_mean,
            'threshold': GZIP_THRESHOLD,
        },
        'timing': {
            'pattern_seconds': round(pattern_time, 3),
            'gzip_seconds': round(gzip_time, 3),
        },
    }
    if think_tag:
        reasoning_mean, reasoning_std = _mean_std(r['reasoning_length']
                                                  for r in records)
        content_mean, content_std = _mean_std(r['content_length']
                                              for r in records)
        res['reasoning_length'] = {
            'mean': reasoning_mean,
            'std': reasoning_std
        }
        res['content_length'] = {'mean': content_mean, 'std': content_std}

    return res


def _sample_with_thresholds(record: Dict[str, Any],
                            stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'model': record['model'],
        'benchmark': record['benchmark'],
        'sample_id': record['sample_id'],
        'prediction_path': record['prediction_path'],
        'length': record['length'],
        'repeat_analysis_skipped': record['repeat_analysis_skipped'],
        'repeat_pattern_ratio': record['repeat_pattern_ratio'],
        'repeat_pattern': record['repeat_pattern'],
        'repeat_pattern_count': record['repeat_pattern_count'],
        'gzip_compression_ratio': record['gzip_compression_ratio'],
        'gzip_compression_threshold':
        stats['gzip_compression_ratio']['threshold'],
    }


def _build_abnormal_samples(
    records: List[Dict[str, Any]],
    benchmark_stats: Dict[str, Any],
    with_reasoning=False,
) -> Dict[str, List[Dict[str, Any]]]:
    abnormal_samples = {
        'repeat_pattern': [],
        'gzip_high_compression': [],
    }
    if with_reasoning:
        abnormal_samples['no_reasoning'] = []
        abnormal_samples['no_reasoning_short'] = []
        abnormal_samples['no_content'] = []

    for record in records:
        stats = benchmark_stats[record['benchmark']]
        sample = _sample_with_thresholds(record, stats)
        if (record['repeat_pattern_count'] >
                stats['repeat_pattern_ratio']['threshold']):
            abnormal_samples['repeat_pattern'].append(sample)
        if (record['gzip_compression_ratio'] >
                stats['gzip_compression_ratio']['threshold']):
            abnormal_samples['gzip_high_compression'].append(sample)
        if with_reasoning:
            if (record['reasoning_length'] == 0):
                abnormal_samples['no_reasoning'].append(sample)
                if record['length'] < stats['length']['p75']:
                    abnormal_samples['no_reasoning_short'].append(sample)
            if (record['content_length'] == 0):
                abnormal_samples['no_content'].append(sample)

    return abnormal_samples


def _fill_abnormal_predictions(
        abnormal_samples: Dict[str, List[Dict[str, Any]]]) -> None:
    samples_by_path = defaultdict(list)
    for samples in abnormal_samples.values():
        for sample in samples:
            samples_by_path[sample['prediction_path']].append(sample)

    for path, samples in samples_by_path.items():
        sample_ids = {sample['sample_id'] for sample in samples}
        predictions = {}
        for sample_id, sample in _iter_prediction_items(Path(path)):
            if sample_id not in sample_ids:
                continue
            if isinstance(sample, dict):
                prediction = sample.get('prediction', '')
            else:
                prediction = sample
            predictions[sample_id] = _prediction_to_text(prediction)
        for sample in samples:
            sample['prediction'] = predictions.get(sample['sample_id'], '')


def _format_analysis_summary_table(report: Dict[str, Any]) -> List[List[str]]:
    with_reasoning = 'reasoning_length' in list(
        report['benchmark_stats'].values())[0]

    if with_reasoning:
        abnormal_counts = defaultdict(
            lambda: {
                'repeat_pattern': 0,
                'gzip_high_compression': 0,
                'no_reasoning': 0,
                'no_reasoning_short': 0,
                'no_content': 0,
            })
    else:
        abnormal_counts = defaultdict(lambda: {
            'repeat_pattern': 0,
            'gzip_high_compression': 0,
        })

    for abnormal_type, samples in report['abnormal_samples'].items():
        for sample in samples:
            abnormal_counts[sample['benchmark']][abnormal_type] += 1

    if with_reasoning:
        table = [[
            'benchmark',
            'num samples',
            'token length (avg / p75 / p90)',
            'reasoning length',
            'content length',
            'pattern rule',
            'gz rule',
            'no reasoning',
            'no reasoning (short)',
            'no content',
        ]]
    else:
        table = [[
            'benchmark',
            'num samples',
            'token length (avg / p75 / p90)',
            'pattern rule',
            'gz rule',
        ]]

    total_n = 0
    total_pattern = 0
    total_gzip = 0
    total_no_reasoning = 0
    total_no_reasoning_short = 0
    total_no_content = 0
    total_length_sum = 0.0
    total_reasoning_sum = 0.0
    total_content_sum = 0.0

    for benchmark, stats in sorted(report['benchmark_stats'].items()):
        counts = abnormal_counts[benchmark]
        n = stats['sample_count']
        length = stats['length']
        mean = int(length['mean'])
        p75 = int(length['p75'])
        p90 = int(length['p90'])

        total_n += n
        total_pattern += counts['repeat_pattern']
        total_gzip += counts['gzip_high_compression']
        total_length_sum += length['mean'] * n
        if with_reasoning:
            total_no_reasoning += counts['no_reasoning']
            total_no_reasoning_short += counts['no_reasoning_short']
            total_no_content += counts['no_content']
            total_reasoning_sum += stats['reasoning_length']['mean'] * n
            total_content_sum += stats['content_length']['mean'] * n

        def _fmt(count, total):
            pct = 100.0 * count / total if total else 0
            return f'{pct:.2f}% ({count})'

        if with_reasoning:
            table.append([
                benchmark,
                str(n),
                f'{mean} / {p75} / {p90}',
                int(stats['reasoning_length']['mean']),
                int(stats['content_length']['mean']),
                _fmt(counts['repeat_pattern'], n),
                _fmt(counts['gzip_high_compression'], n),
                _fmt(counts['no_reasoning'], n),
                _fmt(counts['no_reasoning_short'], n),
                _fmt(counts['no_content'], n),
            ])
        else:
            table.append([
                benchmark,
                str(n),
                f'{mean} / {p75} / {p90}',
                _fmt(counts['repeat_pattern'], n),
                _fmt(counts['gzip_high_compression'], n),
            ])

    def _fmt_overall(count, total):
        pct = 100.0 * count / total if total else 0
        return f'{pct:.2f}% ({count})'

    if with_reasoning:
        table.append([
            'Overall',
            str(total_n),
            str(int(total_length_sum / total_n)) if total_n else '0',
            str(int(total_reasoning_sum / total_n)) if total_n else '0',
            str(int(total_content_sum / total_n)) if total_n else '0',
            _fmt_overall(total_pattern, total_n),
            _fmt_overall(total_gzip, total_n),
            _fmt_overall(total_no_reasoning, total_n),
            _fmt_overall(total_no_reasoning_short, total_n),
            _fmt_overall(total_no_content, total_n),
        ])
    else:
        table.append([
            'Overall',
            str(total_n),
            str(int(total_length_sum / total_n)) if total_n else '0',
            _fmt_overall(total_pattern, total_n),
            _fmt_overall(total_gzip, total_n),
        ])
    return table


def print_analysis_summary(report: Dict[str, Any]) -> None:
    table = _format_analysis_summary_table(report)
    print('\nRepeat analysis summary:\n')
    print(tabulate.tabulate(table, headers='firstrow', floatfmt='.2f'))


def analyze_repeat_predictions(config: ConfigDict,
                               time_str: str = None,
                               output_path: str = None,
                               show_progress: bool = False,
                               print_summary: bool = False,
                               batch_size: int = 512) -> str:
    """Analyze repeated predictions per benchmark.

    For each benchmark:
    1. Batch tokenize to get token lengths (per-benchmark threshold).
    2. Identify candidates above threshold.
    3. Compute pattern/gzip for candidates.
    4. Free tokenize memory before moving to next benchmark.

    Args:
        config: Evaluation config containing models, datasets, work_dir.
        time_str: Timestamp string for the output file name.
        output_path: Optional explicit output path for the report.
        show_progress: Whether to show progress bars.
        print_summary: Whether to print a summary table.
        batch_size: Batch size for tokenization.

    Returns:
        Path to the written report JSON.
    """
    if time_str is None:
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Build tokenizers
    tokenizer_map: Dict[str, TokenizerWrapper] = {}
    for model in config['models']:
        model_abbr = model_abbr_from_cfg(model)
        if model_abbr not in tokenizer_map:
            tokenizer_map[model_abbr] = _build_tokenizer_from_model_cfg(model)

    # Read all predictions grouped by benchmark
    grouped_records, missing_files = _collect_predictions_by_benchmark(config)

    # Analyze each benchmark independently
    all_records: List[Dict[str, Any]] = []
    benchmark_stats: Dict[str, Any] = {}

    for benchmark, records in grouped_records.items():
        if show_progress:
            print(f'\n[{benchmark}] ({len(records)} samples)')

        stats = _analyze_benchmark(records,
                                   tokenizer_map,
                                   show_progress=show_progress,
                                   batch_size=batch_size)
        benchmark_stats[benchmark] = stats
        all_records.extend(records)

    # Build abnormal samples and fill predictions
    abnormal_samples = _build_abnormal_samples(all_records, benchmark_stats)
    _fill_abnormal_predictions(abnormal_samples)

    report = {
        'time': time_str,
        'work_dir': config['work_dir'],
        'settings': {
            'pattern_size':
            PATTERN_SIZE,
            'num_sample_patterns':
            NUM_SAMPLE_PATTERNS,
            'length_unit':
            'tokens',
            'repeat_analysis_threshold':
            'p75 of benchmark token lengths',
            'pattern_rule':
            f'periodic pattern with count > {MIN_PATTERN_COUNT}',
            'gzip_rule':
            f'gzip_compression_ratio > {GZIP_THRESHOLD}',
            'repeat_analysis_min_length_rule':
            'pattern/gzip skipped when length < p75',
        },
        'summary': {
            'model':
            model_abbr_from_cfg(config['models'][0]),
            'num_benchmarks':
            len(benchmark_stats),
            'num_samples':
            len(all_records),
            'num_missing_prediction_files':
            len(missing_files),
            'num_repeat_analysis_skipped':
            sum(r['repeat_analysis_skipped'] for r in all_records),
            'abnormal_counts':
            {key: len(value)
             for key, value in abnormal_samples.items()},
        },
        'missing_prediction_files': missing_files,
        'benchmark_stats': benchmark_stats,
        'abnormal_samples': abnormal_samples,
    }

    if output_path is None:
        output_path = Path(config['work_dir']) / 'summary' / (
            f'repeat_analysis_{time_str}.json')
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=4, ensure_ascii=False),
                           encoding='utf-8')
    if print_summary:
        print_analysis_summary(report)
    for wrapper in tokenizer_map.values():
        wrapper.close()
    _tokenizer_cache.clear()
    return str(output_path)
