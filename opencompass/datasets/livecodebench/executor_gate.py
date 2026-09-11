# Copyright (c) OpenCompass Authors.
"""Refuse LCB / Coding scores when the I/O mock cannot run contest-style code.

CompassBench 2607 loaded a rolled-back ``testing_util.py`` that mocked stdin
with plain ``StringIO``. Legal solutions using ``sys.stdin.buffer`` scored 0
and dropped 20–35 points. The gate lives here, not only in testing_util, so a
stale executor copy still cannot emit ``pass@1``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

LCB_IO_GATE_VERSION = 1
INFRA_EXCEPTION_MAX_RATIO = 0.01
STRINGIO_BUFFER_MARKERS = (
    "'_io.StringIO' object has no attribute 'buffer'",
    '"_io.StringIO" object has no attribute \'buffer\'',
)


class LCBExecutorGateError(RuntimeError):
    """Raised when the LiveCodeBench executor is unsafe to score."""


def _testing_util():
    from . import testing_util
    return testing_util


def testing_util_sha256() -> str:
    path = Path(_testing_util().__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lcb_executor_provenance() -> dict:
    path = Path(_testing_util().__file__).resolve()
    return {
        'io_gate_version': LCB_IO_GATE_VERSION,
        'testing_util_path': str(path),
        'testing_util_sha256': testing_util_sha256(),
    }


def check_lcb_io_executor() -> dict:
    """Run a capability self-test against the loaded testing_util copy."""
    testing_util = _testing_util()
    missing = [
        name for name in ('MockStdinWithBuffer', 'MockBuffer', 'Capturing',
                          'call_method') if not hasattr(testing_util, name)
    ]
    if missing:
        raise LCBExecutorGateError(
            'LCB executor is missing stdin.buffer support '
            f'({", ".join(missing)}). Refusing to emit Coding scores. '
            'Restore MockStdinWithBuffer in '
            'opencompass/datasets/livecodebench/testing_util.py.')

    source = ''
    try:
        source = inspect.getsource(testing_util.call_method)
    except OSError:
        source = ''
    if source and 'MockStdinWithBuffer' not in source:
        raise LCBExecutorGateError(
            'LCB call_method is not using MockStdinWithBuffer; '
            'stdin.buffer solutions would be scored as 0. '
            'Refusing to emit Coding scores.')

    mock = testing_util.MockStdinWithBuffer('first\nsecond\n')
    if not hasattr(mock, 'buffer'):
        raise LCBExecutorGateError(
            'MockStdinWithBuffer has no .buffer attribute. '
            'Refusing to emit Coding scores.')

    first = mock.buffer.readline()
    second = mock.buffer.readline()
    if first != b'first\n' or second != b'second\n':
        raise LCBExecutorGateError(
            'stdin.buffer.readline() is not multiline-safe '
            f'(got {first!r}, {second!r}). Refusing to emit Coding scores.')

    whole = testing_util.MockStdinWithBuffer('alpha\nbeta\n').buffer.read()
    if whole != b'alpha\nbeta\n':
        raise LCBExecutorGateError(f'stdin.buffer.read() returned {whole!r}. '
                                   'Refusing to emit Coding scores.')

    captured_error = None

    def _echo_stdin_buffer():
        import sys
        payload = sys.stdin.buffer.read()
        sys.stdout.buffer.write(payload)

    try:
        with testing_util.Capturing() as output:
            testing_util.call_method(_echo_stdin_buffer, 'hello\nworld\n')
    except Exception as exc:  # noqa: BLE001 - surface any mock regression
        captured_error = exc
        output = []

    if captured_error is not None:
        raise LCBExecutorGateError(
            'LCB I/O self-test crashed while echoing stdin.buffer to '
            f'stdout.buffer: {captured_error!r}. Refusing to emit Coding '
            'scores.')
    if not output or 'hello' not in output[0] or 'world' not in output[0]:
        raise LCBExecutorGateError(
            'LCB I/O self-test did not capture stdout.buffer.write '
            f'(got {output!r}). Refusing to emit Coding scores.')

    provenance = lcb_executor_provenance()
    provenance['io_self_test'] = 'passed'
    return provenance


def _iter_metadata_blobs(final_metadata):
    if not final_metadata:
        return
    for item in final_metadata:
        blobs = item if isinstance(item, list) else [item]
        for blob in blobs:
            if isinstance(blob, dict):
                yield blob
                continue
            if not isinstance(blob, str):
                continue
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                yield {'error': blob, 'error_message': blob}
                continue
            if isinstance(parsed, dict):
                yield parsed
            else:
                yield {'error': blob}


def is_stringio_buffer_failure(metadata: dict) -> bool:
    haystack = ' '.join(
        str(metadata.get(key, ''))
        for key in ('error', 'error_message', 'error_code'))
    return any(marker in haystack for marker in STRINGIO_BUFFER_MARKERS)


def count_stringio_buffer_failures(final_metadata) -> tuple[int, int]:
    total = 0
    hits = 0
    for metadata in _iter_metadata_blobs(final_metadata):
        total += 1
        if is_stringio_buffer_failure(metadata):
            hits += 1
    return total, hits


def assert_lcb_infra_exceptions_within_budget(
        final_metadata, max_ratio: float = INFRA_EXCEPTION_MAX_RATIO) -> dict:
    """Fail closed when the rolled-back StringIO mock shows up in results."""
    total, hits = count_stringio_buffer_failures(final_metadata)
    stats = {
        'checked_runs': total,
        'stringio_buffer_failures': hits,
        'max_ratio': max_ratio,
    }
    if total == 0 or hits == 0:
        return stats
    if hits / total > max_ratio:
        raise LCBExecutorGateError(
            'LCB I/O mock regression detected: '
            f'{hits}/{total} runs failed with StringIO .buffer '
            'AttributeError. Refusing to emit Coding scores. Restore '
            'MockStdinWithBuffer and rescore existing predictions.')
    return stats
