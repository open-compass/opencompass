import filecmp
import json
import os
from typing import Any, List, Optional, Tuple

import fire


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
    """Lines describing where semantic equality fails (shallow key/index level)."""  # noqa: F401, E501
    lines: List[str] = []
    emit = _emit_factory(lines, max_lines, path_prefix)

    if type(left) != type(right):
        emit(f'type mismatch {type(left).__name__} vs {type(right).__name__}')
        return lines

    if isinstance(left, dict):
        for k in _sort_keys_for_report(set(left) | set(right)):
            if len(lines) >= max_lines:
                break
            if k not in left:
                emit(f'key {k!r} only in {label2}')
            elif k not in right:
                emit(f'key {k!r} only in {label1}')
            elif not _semantic_equal(left[k], right[k]):
                emit(f'key {k!r} differs (semantic)')
        return lines

    if isinstance(left, list):
        if len(left) != len(right):
            emit(f'list len {len(left)} ({label1}) vs {len(right)} ({label2})')
        for i in range(min(len(left), len(right))):
            if len(lines) >= max_lines:
                break
            if not _semantic_equal(left[i], right[i]):
                emit(f'index {i} differs (semantic)')
        if len(left) > len(right) and len(lines) < max_lines:
            emit(f'indices {len(right)}..{len(left) - 1} only in {label1}')
        elif len(right) > len(left) and len(lines) < max_lines:
            emit(f'indices {len(left)}..{len(right) - 1} only in {label2}')
        return lines

    emit('scalar / leaf value differs under canonicalization')
    return lines


def _json_pair_compare_reason(
    path1: str,
    path2: str,
    rel_path: str,
    json_diff_max_lines: int,
) -> Optional[str]:
    """None if pair matches; else multi-line reason (with header). Loads each file once."""  # noqa: F401, E501
    try:
        left = _load_json(path1)
        right = _load_json(path2)
    except json.JSONDecodeError as e:
        return f'Invalid JSON: {e}'
    except OSError as e:
        return f'Could not read JSON file: {e}'

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
        return header + '\n'.join(detail)

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
    return 'JSON semantic mismatch. Details:\n' + '\n'.join(detail)


def _is_json_file(name: str) -> bool:
    """True if filename should be parsed as JSON."""
    return name.lower().endswith('.json')


def compare_results(
    folder1: str,
    folder2: str,
    compare_type: str = 'predictions',
    results_ignore_list: Optional[list] = None,
    raise_on_diff: bool = True,
    json_diff_max_lines: int = 10,
) -> Optional[List[Tuple[str, str]]]:
    """Entry for scripts: take first subpath under each root, then compare compare_type."""  # noqa: F401, E501
    if results_ignore_list is None:
        results_ignore_list = [
            'srbench.json',
            'dingo_en_192.json',
            'dingo_zh_170.json',
            'qa_dingo_cn.json',
        ]

    assert os.path.isdir(folder1), f'Folder does not exist: {folder1}'
    assert os.path.isdir(folder2), f'Folder does not exist: {folder2}'

    sub_folder1 = get_all_subpaths(folder1)[0]
    sub_folder2 = get_all_subpaths(folder2)[0]

    print(f'compare {compare_type}')
    return compare_folders(
        os.path.join(sub_folder1, compare_type),
        os.path.join(sub_folder2, compare_type),
        results_ignore_list=results_ignore_list,
        raise_on_diff=raise_on_diff,
        json_diff_max_lines=json_diff_max_lines,
    )


def compare_folders(
    folder1: str,
    folder2: str,
    results_ignore_list: Optional[list] = None,
    raise_on_diff: bool = True,
    json_diff_max_lines: int = 10,
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
                diff_files.append((rel_path, 'File missing in second folder'))
                continue

            if _is_json_file(file):
                reason = _json_pair_compare_reason(path1, path2, rel_path,
                                                   json_diff_max_lines)
                if reason is not None:
                    diff_files.append((rel_path, reason))
            elif not filecmp.cmp(path1, path2, shallow=False):
                diff_files.append((rel_path, 'Content differs'))

    for root, _dirs, files in os.walk(folder2):
        for file in files:
            if file in results_ignore_list:
                continue
            rel_path = os.path.relpath(os.path.join(root, file), folder2)
            path1 = os.path.join(folder1, rel_path)
            if not os.path.exists(path1):
                diff_files.append((rel_path, 'File missing in first folder'))

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


if __name__ == '__main__':
    fire.Fire()
