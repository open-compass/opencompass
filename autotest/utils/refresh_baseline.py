"""Backup and refresh one mock-api baseline func_type from a good run."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional


def _run_compare(src: str, dst: str, compare_type: str) -> None:
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'compare_results.py'),
        'compare_results',
        src,
        dst,
        compare_type,
    ]
    print('+', ' '.join(cmd))
    subprocess.check_call(cmd)


def refresh_baseline(
    src: str,
    dst: str,
    backup: bool = True,
    verify: bool = True,
) -> Optional[str]:
    assert os.path.isdir(src), f'Source missing: {src}'
    ts = time.strftime('%Y%m%d_%H%M%S')
    bak: Optional[str] = None
    if backup and os.path.isdir(dst):
        bak = f'{dst}.bak_{ts}'
        print(f'Backup: {dst} -> {bak}')
        shutil.copytree(dst, bak)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    # copy contents so dst keeps a single workdir layout from src
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    print(f'Refreshed baseline: {dst} from {src}')
    if verify:
        for kind in ('predictions', 'results', 'summary'):
            _run_compare(src, dst, kind)
        print('Compare verify passed.')
    return bak


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('src',
                   help='Good run dir: REPORT_ROOT/<run_id>/<func_type>')
    p.add_argument(
        'dst', help='Baseline dir: REPORT_ROOT/mock-api-baseline/<func_type>')
    p.add_argument('--no-backup', action='store_true')
    p.add_argument('--no-verify', action='store_true')
    args = p.parse_args(argv)
    refresh_baseline(
        args.src,
        args.dst,
        backup=not args.no_backup,
        verify=not args.no_verify,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
