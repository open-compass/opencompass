"""Validate cmd_test regression summary scores are within [min, max]."""
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple


def _latest_summary_csv(result_root: str) -> str:
    pattern = os.path.join(result_root, '*', 'summary', '*.csv')
    matches = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if not matches:
        raise FileNotFoundError(
            f'No summary CSV under {result_root}/*/summary/*.csv')
    matches.sort(key=lambda p: (os.path.getmtime(p), p))
    return matches[-1]


def _last_field_score(csv_path: str) -> float:
    with open(csv_path, encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f'Empty summary CSV: {csv_path}')
    last = lines[-1]
    field = last.split(',')[-1].strip()
    try:
        return float(field)
    except ValueError as e:
        raise ValueError(
            f'Cannot parse score from last field {field!r} in {csv_path}'
        ) from e


def check_cmd_scores(
    run_dir: str,
    score_min: float = 75.0,
    score_max: float = 85.0,
    result_count: int = 3,
) -> List[Tuple[str, float]]:
    """Return [(csv_path, score), ...] or raise AssertionError."""
    assert os.path.isdir(run_dir), f'Run dir missing: {run_dir}'
    checked: List[Tuple[str, float]] = []
    for idx in range(1, result_count + 1):
        result_root = os.path.join(run_dir, f'regression_result{idx}')
        if not os.path.isdir(result_root):
            raise AssertionError(f'Missing {result_root}')
        csv_path = _latest_summary_csv(result_root)
        score = _last_field_score(csv_path)
        checked.append((csv_path, score))
        ok = score_min <= score <= score_max
        print(f'regression_result{idx}: score={score} file={csv_path}')
        if not ok:
            raise AssertionError(
                f'score {score} not in [{score_min}, {score_max}] '
                f'(file={csv_path})')
    return checked


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir',
                   help='REPORT_ROOT/<run_id> with regression_result*')
    p.add_argument('--min', dest='score_min', type=float, default=75.0)
    p.add_argument('--max', dest='score_max', type=float, default=85.0)
    p.add_argument('--count', type=int, default=3)
    args = p.parse_args(argv)
    check_cmd_scores(
        args.run_dir,
        score_min=args.score_min,
        score_max=args.score_max,
        result_count=args.count,
    )
    print('All cmd_test scores within range.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
