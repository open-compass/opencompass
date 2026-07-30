import difflib
import filecmp
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import fire

_SUMMARY_TS_RE = re.compile(r'summary_(\d{8}_\d{6})', re.IGNORECASE)
_SUMMARY_COMPARE_EXTS = ('csv', 'md')
_LCB_PATH_MARKERS = ('lcb', 'livecodebench', 'livecodebench_pro')
_TRACEBACK_OBJ_RE = re.compile(r'<traceback object at 0x[0-9a-fA-F]+>')
_MEM_ADDR_RE = re.compile(r'0x[0-9a-fA-F]+')
_VALUE_PREVIEW_MAX = 512
_TEXT_DIFF_MAX_LINES = 2


def _file_pair_abs_paths(path1: str, path2: str) -> str:
    """Absolute paths for a compared pair (for manual inspection)."""
    return (f'  left:  {os.path.abspath(path1)}\n'
            f'  right: {os.path.abspath(path2)}')


def _with_file_paths(message: str, path1: str, path2: str) -> str:
    """Append absolute paths of both files under a diff/mismatch message."""
    return f'{message}\n{_file_pair_abs_paths(path1, path2)}'


def _load_json(path: str) -> Any:
    """Load UTF-8 JSON from disk."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _canonicalize(obj: Any) -> Any:
    """Normalize JSON for comparison (sorted dict keys; sortable lists sorted)."""  # noqa: F401, E501
    if isinstance(obj, dict):
        return tuple(sorted(
            (k, _canonicalize(v)) for k, v in obj.items()))  # noqa: F401, E501
    if isinstance(obj, list):
        elems = [_canonicalize(x) for x in obj]
        try:
            return tuple(sorted(elems))
        except TypeError:
            return tuple(elems)
    return obj


def _semantic_equal(a: Any, b: Any) -> bool:
    """True if canonicalized a and b are equal."""
    return _canonicalize(a) == _canonicalize(b)


def _path_uses_do_sample(rel_path: str) -> bool:
    """True if rel_path triggers key-shape-only JSON compare (ignore leaf values)."""  # noqa: F401, E501
    normalized = rel_path.lower().replace('-', '_')
    return 'do_sample' in normalized


def _path_is_lcb_results(rel_path: str, compare_type: str) -> bool:
    if compare_type != 'results':
        return False
    normalized = rel_path.lower().replace('-', '_')
    return any(marker in normalized for marker in _LCB_PATH_MARKERS)


def _normalize_trace_text(text: str) -> str:
    text = _TRACEBACK_OBJ_RE.sub('<traceback>', text)
    return _MEM_ADDR_RE.sub('0x0', text)


def _normalize_final_metadata(value: Any) -> Any:
    """Drop volatile traceback addresses from LCB final_metadata blobs."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('{'):
            try:
                return _normalize_final_metadata(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return _normalize_trace_text(value)
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if key == 'error' and isinstance(item, str):
                normalized[key] = _normalize_trace_text(item)
            elif key in ('error_code', 'error_message'):
                normalized[key] = item
            else:
                normalized[key] = _normalize_final_metadata(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_final_metadata(item) for item in value]
    return value


def _normalize_lcb_results(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: (_normalize_final_metadata(val)
                  if key == 'final_metadata' else _normalize_lcb_results(val))
            for key, val in obj.items()
        }
    if isinstance(obj, list):
        return [_normalize_lcb_results(item) for item in obj]
    return obj


def _sort_keys_for_report(keys):
    """Sort keys: numeric strings by int value, else lexically."""

    def key_fn(k):
        s = str(k)
        return (0, int(s)) if s.isdigit() else (1, s)

    return sorted(keys, key=key_fn)


def _emit_factory(lines: List[str], max_lines: int, path_prefix: str):
    """Return emit(msg) that appends to lines, capped at max_lines."""

    def emit(msg: str) -> None:
        if len(lines) < max_lines:
            lines.append(f'{path_prefix}: {msg}' if path_prefix else msg)

    return emit


def _preview_value(value: Any, max_len: int = _VALUE_PREVIEW_MAX) -> str:
    """Compact single-line preview of a value for diff messages."""
    if isinstance(value, str):
        text = value.replace('\n', '\\n').replace('\r', '\\r')
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            text = repr(value)
        text = text.replace('\n', ' ')
    if len(text) > max_len:
        return text[:max_len - 3] + '...'
    return text


def _append_value_diff(
    lines: List[str],
    max_lines: int,
    path: str,
    left: Any,
    right: Any,
    note: str = 'value differs',
) -> None:
    """Append path note plus up to two preview lines (-left / +right)."""
    if len(lines) >= max_lines:
        return
    lines.append(f'{path}: {note}' if path else note)
    if len(lines) < max_lines:
        lines.append('  - ' + _preview_value(left))
    if len(lines) < max_lines:
        lines.append('  + ' + _preview_value(right))


def _text_diff_snippet(
    path1: str,
    path2: str,
    max_diff_lines: int = _TEXT_DIFF_MAX_LINES,
) -> str:
    """Short unified-diff style snippet for two text files (1-2 +/- lines)."""
    try:
        with open(path1, encoding='utf-8', errors='replace') as f1:
            lines1 = f1.read().splitlines()
        with open(path2, encoding='utf-8', errors='replace') as f2:
            lines2 = f2.read().splitlines()
    except OSError as e:
        return _with_file_paths(
            f'Content differs (could not read for diff: {e})', path1, path2)

    if lines1 == lines2:
        return _with_file_paths(
            'Content differs (byte-level; text lines equal)', path1, path2)

    diff_lines: List[str] = []
    for line in difflib.unified_diff(lines1, lines2, lineterm='', n=0):
        if line.startswith(('+++', '---', '@@')):
            continue
        if line.startswith(('+', '-')):
            preview = line if len(line) <= _VALUE_PREVIEW_MAX else (
                line[:_VALUE_PREVIEW_MAX - 3] + '...')
            diff_lines.append(preview)
            if len(diff_lines) >= max_diff_lines:
                break
    if not diff_lines:
        return _with_file_paths('Content differs', path1, path2)
    return _with_file_paths('Content differs:\n' + '\n'.join(diff_lines),
                            path1, path2)


def _key_structure_equal(a: Any, b: Any) -> bool:
    """True if types, dict keys, and list lengths match recursively; ignore leaves."""  # noqa: F401, E501
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        if set(a) != set(b):
            return False
        return all(_key_structure_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_key_structure_equal(a[i], b[i]) for i in range(len(a)))
    return True


def _key_structure_diff_lines(
    left: Any,
    right: Any,
    max_lines: int = 10,
    path_prefix: str = '',
    label1: str = 'file1',
    label2: str = 'file2',
) -> List[str]:
    """Human-readable structure diffs only (keys/lengths/types; not values)."""  # noqa: F401, E501
    lines: List[str] = []
    emit = _emit_factory(lines, max_lines, path_prefix)

    if type(left) != type(right):
        emit(f'type mismatch {type(left).__name__} vs {type(right).__name__}')
        return lines

    if isinstance(left, dict):
        for k in _sort_keys_for_report(set(left) | set(right)):
            if len(lines) >= max_lines:
                break
            sub = f'{path_prefix}.{k}' if path_prefix else str(k)
            if k not in left:
                emit(f'key {k!r} only in {label2}')
            elif k not in right:
                emit(f'key {k!r} only in {label1}')
            else:
                lines.extend(
                    _key_structure_diff_lines(
                        left[k],
                        right[k],
                        max_lines=max_lines - len(lines),
                        path_prefix=sub,
                        label1=label1,
                        label2=label2,
                    ))
        return lines

    if isinstance(left, list):
        if len(left) != len(right):
            emit(f'list len {len(left)} ({label1}) vs {len(right)} ({label2})')
        for i in range(min(len(left), len(right))):
            if len(lines) >= max_lines:
                break
            sub = f'{path_prefix}[{i}]' if path_prefix else f'[{i}]'
            lines.extend(
                _key_structure_diff_lines(
                    left[i],
                    right[i],
                    max_lines=max_lines - len(lines),
                    path_prefix=sub,
                    label1=label1,
                    label2=label2,
                ))
        return lines

    return lines


def json_semantic_diff_lines(
    left: Any,
    right: Any,
    max_lines: int = 10,
    path_prefix: str = '',
    label1: str = 'file1',
    label2: str = 'file2',
) -> List[str]:
    """Lines describing where semantic equality fails (recursive paths)."""  # noqa: F401, E501
    lines: List[str] = []
    emit = _emit_factory(lines, max_lines, path_prefix)

    if type(left) != type(right):
        emit(f'type mismatch {type(left).__name__} vs {type(right).__name__}')
        if len(lines) < max_lines:
            lines.append('  - ' + _preview_value(left))
        if len(lines) < max_lines:
            lines.append('  + ' + _preview_value(right))
        return lines

    if isinstance(left, dict):
        for k in _sort_keys_for_report(set(left) | set(right)):
            if len(lines) >= max_lines:
                break
            sub = f'{path_prefix}.{k}' if path_prefix else str(k)
            if k not in left:
                emit(f'key {k!r} only in {label2}')
            elif k not in right:
                emit(f'key {k!r} only in {label1}')
            elif not _semantic_equal(left[k], right[k]):
                if isinstance(left[k], (dict, list)) and isinstance(
                        right[k], (dict, list)):
                    before = len(lines)
                    lines.extend(
                        json_semantic_diff_lines(
                            left[k],
                            right[k],
                            max_lines=max_lines - len(lines),
                            path_prefix=sub,
                            label1=label1,
                            label2=label2,
                        ))
                    if len(lines) == before:
                        _append_value_diff(lines, max_lines, sub, left[k],
                                           right[k])
                else:
                    _append_value_diff(lines, max_lines, sub, left[k],
                                       right[k])
        return lines

    if isinstance(left, list):
        if len(left) != len(right):
            emit(f'list len {len(left)} ({label1}) vs {len(right)} ({label2})')
        for i in range(min(len(left), len(right))):
            if len(lines) >= max_lines:
                break
            sub = f'{path_prefix}[{i}]' if path_prefix else f'[{i}]'
            if not _semantic_equal(left[i], right[i]):
                if isinstance(left[i], (dict, list)) and isinstance(
                        right[i], (dict, list)):
                    before = len(lines)
                    lines.extend(
                        json_semantic_diff_lines(
                            left[i],
                            right[i],
                            max_lines=max_lines - len(lines),
                            path_prefix=sub,
                            label1=label1,
                            label2=label2,
                        ))
                    if len(lines) == before:
                        _append_value_diff(lines, max_lines, sub, left[i],
                                           right[i])
                else:
                    _append_value_diff(lines, max_lines, sub, left[i],
                                       right[i])
        if len(left) > len(right) and len(lines) < max_lines:
            emit(f'indices {len(right)}..{len(left) - 1} only in {label1}')
        elif len(right) > len(left) and len(lines) < max_lines:
            emit(f'indices {len(left)}..{len(right) - 1} only in {label2}')
        return lines

    _append_value_diff(
        lines,
        max_lines,
        path_prefix,
        left,
        right,
        note='scalar / leaf value differs',
    )
    return lines


def _json_pair_compare_reason(
    path1: str,
    path2: str,
    rel_path: str,
    json_diff_max_lines: int,
    compare_type: str = '',
) -> Optional[str]:
    """None if pair matches; else multi-line reason (with header). Loads each file once."""  # noqa: F401, E501
    try:
        left = _load_json(path1)
        right = _load_json(path2)
    except json.JSONDecodeError as e:
        return _with_file_paths(f'Invalid JSON: {e}', path1, path2)
    except OSError as e:
        return _with_file_paths(f'Could not read JSON file: {e}', path1, path2)

    if _path_is_lcb_results(rel_path, compare_type):
        left = _normalize_lcb_results(left)
        right = _normalize_lcb_results(right)

    label1, label2 = os.path.basename(path1), os.path.basename(path2)
    do_sample = _path_uses_do_sample(rel_path)

    if do_sample:
        if _key_structure_equal(left, right):
            return None
        detail = _key_structure_diff_lines(
            left,
            right,
            max_lines=json_diff_max_lines,
            path_prefix='',
            label1=label1,
            label2=label2,
        )
        if not detail:
            detail = ['(no structural diff lines; unexpected mismatch)']
        header = (
            'JSON key/shape mismatch (do_sample path; leaf values ignored). '
            'Details:\n')
        return _with_file_paths(header + '\n'.join(detail), path1, path2)

    if _semantic_equal(left, right):
        return None
    detail = json_semantic_diff_lines(
        left,
        right,
        max_lines=json_diff_max_lines,
        path_prefix='',
        label1=label1,
        label2=label2,
    )
    if not detail:
        detail = [
            '(no per-key diff lines; structure may be non-dict/non-list)'
        ]
    return _with_file_paths(
        'JSON semantic mismatch. Details:\n' + '\n'.join(detail),
        path1,
        path2,
    )


def _is_json_file(name: str) -> bool:
    """True if filename should be parsed as JSON."""
    return name.lower().endswith('.json')


def _summary_file_sort_key(filename: str, filepath: str) -> Tuple[str, float]:
    """Sort key: embedded summary timestamp, else file mtime."""
    match = _SUMMARY_TS_RE.search(filename)
    if match:
        return (match.group(1), 0.0)
    return ('', os.path.getmtime(filepath))


def _latest_summary_files_by_dir(root: str) -> Dict[str, Dict[str, str]]:
    """Per relative dir, pick newest summary_*.{csv,md} by timestamp suffix."""
    latest: Dict[str, Dict[str, Tuple[Tuple[str, float],
                                      str]]] = defaultdict(dict)
    for dirpath, _dirs, files in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == '.':
            rel_dir = ''
        for name in files:
            lower = name.lower()
            if not lower.startswith('summary_'):
                continue
            ext = lower.rsplit('.', 1)[-1] if '.' in lower else ''
            if ext not in _SUMMARY_COMPARE_EXTS:
                continue
            full_path = os.path.join(dirpath, name)
            sort_key = _summary_file_sort_key(name, full_path)
            prev = latest[rel_dir].get(ext)
            if prev is None or sort_key > prev[0]:
                latest[rel_dir][ext] = (sort_key, full_path)
    return {
        rel_dir: {ext: path
                  for ext, (_key, path) in exts.items()}
        for rel_dir, exts in latest.items()
    }


def compare_summary_folders(
    folder1: str,
    folder2: str,
    raise_on_diff: bool = True,
) -> Optional[List[Tuple[str, str]]]:
    """Compare only the newest summary_*.csv and summary_*.md in subdir."""
    assert os.path.isdir(folder1), f'Folder does not exist: {folder1}'
    assert os.path.isdir(folder2), f'Folder does not exist: {folder2}'

    latest1 = _latest_summary_files_by_dir(folder1)
    latest2 = _latest_summary_files_by_dir(folder2)
    all_dirs = sorted(set(latest1) | set(latest2))

    diff_files: List[Tuple[str, str]] = []
    for rel_dir in all_dirs:
        dir_label = rel_dir or '.'
        files1 = latest1.get(rel_dir, {})
        files2 = latest2.get(rel_dir, {})
        for ext in _SUMMARY_COMPARE_EXTS:
            path1 = files1.get(ext)
            path2 = files2.get(ext)
            rel_name = f'{dir_label}/latest.summary.{ext}'
            if path1 is None and path2 is None:
                continue
            if path1 is None:
                diff_files.append((
                    rel_name,
                    _with_file_paths(
                        f'No summary_*.{ext} in first folder',
                        os.path.join(folder1, rel_dir or '.'),
                        path2,
                    ),
                ))
                continue
            if path2 is None:
                diff_files.append((
                    rel_name,
                    _with_file_paths(
                        f'No summary_*.{ext} in second folder',
                        path1,
                        os.path.join(folder2, rel_dir or '.'),
                    ),
                ))
                continue
            if not filecmp.cmp(path1, path2, shallow=False):
                snippet = _text_diff_snippet(path1, path2)
                diff_files.append((rel_name, snippet))

    if diff_files:
        header = (
            'Summary compare uses newest summary_*.{csv,md} per directory; '
            'timestamped .txt and older files are ignored.\n')
        error_msg = header + 'Found differences:\n' + '\n'.join(
            f'{path}: {reason}' for path, reason in diff_files)
        if raise_on_diff:
            raise AssertionError(error_msg)
        return diff_files
    return [] if not raise_on_diff else None


def compare_results(
    folder1: str,
    folder2: str,
    compare_type: str = 'predictions',
    results_ignore_list: Optional[list] = None,
    raise_on_diff: bool = True,
    json_diff_max_lines: int = 10,
) -> Optional[List[Tuple[str, str]]]:
    """Pick a stable workdir under each root, then compare compare_type."""
    if results_ignore_list is None:
        results_ignore_list = ['srbench.json']

    assert os.path.isdir(folder1), f'Folder does not exist: {folder1}'
    assert os.path.isdir(folder2), f'Folder does not exist: {folder2}'

    sub_folder1 = pick_compare_workdir(folder1)
    sub_folder2 = pick_compare_workdir(folder2)
    print(f'compare {compare_type}')
    print(f'  workdir1: {sub_folder1}')
    print(f'  workdir2: {sub_folder2}')
    target1 = os.path.join(sub_folder1, compare_type)
    target2 = os.path.join(sub_folder2, compare_type)
    if compare_type == 'summary':
        return compare_summary_folders(
            target1,
            target2,
            raise_on_diff=raise_on_diff,
        )
    return compare_folders(
        target1,
        target2,
        results_ignore_list=results_ignore_list,
        raise_on_diff=raise_on_diff,
        json_diff_max_lines=json_diff_max_lines,
        compare_type=compare_type,
    )


def compare_folders(
    folder1: str,
    folder2: str,
    results_ignore_list: Optional[list] = None,
    raise_on_diff: bool = True,
    json_diff_max_lines: int = 10,
    compare_type: str = '',
) -> Optional[List[Tuple[str, str]]]:
    """
    Walk both trees; same rel_path must match (JSON per module rules, else binary). # noqa: F401, E501

    If raise_on_diff: raise AssertionError on mismatch; else return list of
    (rel_path, reason), or [] if equal.
    """
    if results_ignore_list is None:
        results_ignore_list = []

    assert os.path.isdir(folder1), f'Folder does not exist: {folder1}'
    assert os.path.isdir(folder2), f'Folder does not exist: {folder2}'

    diff_files: List[Tuple[str, str]] = []

    for root, _dirs, files in os.walk(folder1):
        for file in files:
            basename = os.path.basename(file)
            if basename in results_ignore_list:
                print(f'ignore case: {basename}')
                continue

            rel_path = os.path.relpath(os.path.join(root, file), folder1)
            path1 = os.path.join(root, file)
            path2 = os.path.join(folder2, rel_path)

            if not os.path.exists(path2):
                diff_files.append((
                    rel_path,
                    _with_file_paths('File missing in second folder', path1,
                                     path2),
                ))
                continue

            if _is_json_file(file):
                reason = _json_pair_compare_reason(
                    path1,
                    path2,
                    rel_path,
                    json_diff_max_lines,
                    compare_type=compare_type,
                )
                if reason is not None:
                    diff_files.append((rel_path, reason))
            elif not filecmp.cmp(path1, path2, shallow=False):
                diff_files.append((rel_path, _text_diff_snippet(path1, path2)))

    for root, _dirs, files in os.walk(folder2):
        for file in files:
            if file in results_ignore_list:
                continue
            rel_path = os.path.relpath(os.path.join(root, file), folder2)
            path1 = os.path.join(folder1, rel_path)
            path2 = os.path.join(root, file)
            if not os.path.exists(path1):
                diff_files.append((
                    rel_path,
                    _with_file_paths('File missing in first folder', path1,
                                     path2),
                ))

    if diff_files:
        error_msg = 'Found differences in files:\n' + '\n'.join(
            f'{path}: {reason}' for path, reason in diff_files)
        if raise_on_diff:
            raise AssertionError(error_msg)
        return diff_files
    return [] if not raise_on_diff else None


def get_all_subpaths(directory: str) -> List[str]:
    """os.walk order: dirs then files under each root (full paths)."""
    if not os.path.isdir(directory):
        raise ValueError(f'Directory does not exist: {directory}')

    subpaths: List[str] = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            subpaths.append(os.path.join(root, dir_name))
        for file_name in files:
            subpaths.append(os.path.join(root, file_name))
    return subpaths


_WORKDIR_MARKERS = ('predictions', 'results', 'summary')


def _has_compare_markers(path: str) -> bool:
    return any(
        os.path.isdir(os.path.join(path, marker))
        for marker in _WORKDIR_MARKERS)


def pick_compare_workdir(root: str) -> str:
    """Pick OpenCompass workdir under root for predictions/results/summary.

    Prefers immediate child dirs that contain predictions/results/summary.
    If several match, choose the newest by mtime. If root itself already has
    markers, use root. Falls back to newest immediate subdirectory.
    """
    if not os.path.isdir(root):
        raise ValueError(f'Directory does not exist: {root}')
    if _has_compare_markers(root):
        return root

    children = [
        os.path.join(root, name) for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)) and not name.startswith('.')
        and not name.endswith('.bak') and '.bak_' not in name
    ]
    marked = [p for p in children if _has_compare_markers(p)]
    pool = marked or children
    if not pool:
        raise ValueError(f'No workdir with {_WORKDIR_MARKERS} under {root}')
    pool.sort(key=lambda p: (os.path.getmtime(p), p))
    return pool[-1]


if __name__ == '__main__':
    fire.Fire()
