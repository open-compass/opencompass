import filecmp
import json
import os

import fire


def _canonicalize(obj):
    """Normalize JSON-like structures for equality.

    - dict: key order ignored (sorted by key).
    - list: element order ignored when elements are mutually sortable after
      canonicalization; otherwise order is preserved.
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, _canonicalize(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        elems = [_canonicalize(x) for x in obj]
        try:
            return tuple(sorted(elems))
        except TypeError:
            return tuple(elems)
    return obj


def _json_files_semantically_equal(path1, path2):
    with open(path1, encoding='utf-8') as f1, open(path2,
                                                   encoding='utf-8') as f2:
        left = json.load(f1)
        right = json.load(f2)
    return _canonicalize(left) == _canonicalize(right)


def _semantic_equal(a, b):
    return _canonicalize(a) == _canonicalize(b)


def _sort_keys_for_report(keys):
    """Stable order: numeric string keys first (0,1,2…), then others by str."""

    def key_fn(k):
        s = str(k)
        return (0, int(s)) if s.isdigit() else (1, s)

    return sorted(keys, key=key_fn)


def json_semantic_diff_lines(left,
                             right,
                             max_lines=10,
                             path_prefix='',
                             label1='file1',
                             label2='file2'):
    """Describe semantic inequality between two decoded JSON values.

    OpenCompass prediction files are usually dict[str, record]. Reports
    which keys / list indices differ under the same canonicalization as
    :func:`_canonicalize`.
    """
    lines = []

    def emit(msg):
        if len(lines) < max_lines:
            lines.append(f'{path_prefix}: {msg}' if path_prefix else msg)

    if type(left) != type(right):
        emit(f'type mismatch {type(left).__name__} vs {type(right).__name__}')
        return lines

    if isinstance(left, dict):
        all_keys = set(left) | set(right)
        for k in _sort_keys_for_report(all_keys):
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
        n = min(len(left), len(right))
        for i in range(n):
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


def json_file_semantic_diff_report(path1, path2, max_lines=200):
    """
    Load two JSON files and return lines pointing to differing records/keys.
    """
    label1 = os.path.basename(path1)
    label2 = os.path.basename(path2)
    with open(path1, encoding='utf-8') as f1, open(path2,
                                                   encoding='utf-8') as f2:
        left = json.load(f1)
        right = json.load(f2)
    detail = json_semantic_diff_lines(
        left,
        right,
        max_lines=max_lines,
        path_prefix='',
        label1=label1,
        label2=label2,
    )
    if not detail:
        detail = [
            '(no per-key diff lines; structure may be non-dict/non-list)'
        ]
    return detail


def _is_json_file(name):
    return name.lower().endswith('.json')


def compare_results(folder1,
                    folder2,
                    compare_type: str = 'predictions',
                    results_ignore_list: list = [
                        'srbench.json', 'dingo_en_192.json',
                        'dingo_zh_170.json', 'qa_dingo_cn.json', 'srbench.json'
                    ],
                    raise_on_diff: bool = True,
                    json_diff_max_lines: int = 10):
    # Initialize ignore_list if not provided
    if results_ignore_list is None:
        results_ignore_list = []

    # Verify that both folders exist
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


def compare_folders(folder1,
                    folder2,
                    results_ignore_list=None,
                    raise_on_diff: bool = True,
                    json_diff_max_lines: int = 10):
    '''
    Compare the contents of files with the same name in two folders
    and their subfolders
    '''
    # Initialize ignore_list if not provided
    if results_ignore_list is None:
        results_ignore_list = []

    # Verify that both folders exist
    assert os.path.isdir(folder1), f'Folder does not exist: {folder1}'
    assert os.path.isdir(folder2), f'Folder does not exist: {folder2}'

    # List to store differences found
    diff_files = []

    # Walk through the first folder and compare files
    for root, dirs, files in os.walk(folder1):
        for file in files:
            # Skip files in the ignore list
            if os.path.basename(file) in results_ignore_list:
                print('ignore case: ' + os.path.basename(file))
                continue

            # Get relative path to maintain folder structure
            rel_path = os.path.relpath(os.path.join(root, file), folder1)
            file2 = os.path.join(folder2, rel_path)

            # Check if file exists in the second folder
            if not os.path.exists(file2):
                diff_files.append((rel_path, 'File missing in second folder'))
                continue

            path1 = os.path.join(root, file)
            if _is_json_file(file):
                try:
                    same = _json_files_semantically_equal(path1, file2)
                except json.JSONDecodeError as e:
                    diff_files.append((rel_path, f'Invalid JSON: {e}'))
                    continue
                if not same:
                    try:
                        detail_lines = json_file_semantic_diff_report(
                            path1, file2, max_lines=json_diff_max_lines)
                    except (json.JSONDecodeError, OSError) as e:
                        detail_lines = [f'(could not diff: {e})']
                    reason = ('JSON content differs (semantic). Details:\n' +
                              '\n'.join(detail_lines))
                    diff_files.append((rel_path, reason))
            elif not filecmp.cmp(path1, file2, shallow=False):
                diff_files.append((rel_path, 'Content differs'))

    # Check for files in folder2 that don't exist in folder1
    for root, dirs, files in os.walk(folder2):
        for file in files:
            # Skip files in the ignore list
            if file in results_ignore_list:
                continue

            rel_path = os.path.relpath(os.path.join(root, file), folder2)
            file1 = os.path.join(folder1, rel_path)
            if not os.path.exists(file1):
                diff_files.append((rel_path, 'File missing in first folder'))

    # Raise or return when differences were found
    if diff_files:
        error_msg = 'Found differences in files:\n'
        error_msg += '\n'.join(f'{path}: {reason}'
                               for path, reason in diff_files)
        if raise_on_diff:
            raise AssertionError(error_msg)
        return diff_files
    return [] if not raise_on_diff else None


def get_all_subpaths(directory):
    '''
    Get all subpaths (files and directories) within a given directory.
    Args:
        directory (str): The root directory path to search
    Returns:
        list: A list of all complete subpaths
    '''
    subpaths = []

    # Verify the directory exists
    if not os.path.isdir(directory):
        raise ValueError(f'Directory does not exist: {directory}')

    # Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        # Add directories first
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            subpaths.append(full_path)

        # Then add files
        for file_name in files:
            full_path = os.path.join(root, file_name)
            subpaths.append(full_path)

    return subpaths


if __name__ == '__main__':
    fire.Fire()
