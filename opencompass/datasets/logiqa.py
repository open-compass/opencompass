import hashlib
import os
import os.path as osp
import re

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_logger

from .base import BaseDataset

# Original release: https://github.com/lgw863/LogiQA-dataset
_COMMIT = 'ff6c4cbca47627b3ac2da94a29fa28204a167b41'
_URL = f'https://raw.githubusercontent.com/lgw863/LogiQA-dataset/{_COMMIT}'
_FILES = {
    'en': {
        'train': ('Train.txt', '6d388437e81dd9513362ddf1c0cb2941'),
        'validation': ('Eval.txt', 'dd481b2614a82085f2f6b933c5e86650'),
        'test': ('Test.txt', '7cb64d0ac374c59b4ef5699ba23e0d1c'),
    },
    'zh': {
        'train': ('zh_train.txt', 'b085290e704a3a5e9ec11585c606d008'),
        'validation': ('zh_eval.txt', 'a676fe3b1e35998de444591fc6ac83ad'),
        'test': ('zh_test.txt', 'f4d5d70af34e50ad7c1cb7ad316e1fbe'),
    },
}
# HF mirrors used by earlier configs. Their loading scripts drop the first
# character of many options, so we read the original files instead.
_LEGACY_IDS = {'lucasmccabe/logiqa': 'en', 'jiacheng-ye/logiqa-zh': 'zh'}
_DEFAULT_ROOT = './data/LogiQA'
_SEP = '.．,，:：、。'


def _md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _fetch(root, file_name, md5):
    import requests

    path = osp.join(root, file_name)
    if osp.exists(path):
        return path
    os.makedirs(root, exist_ok=True)
    url = f'{_URL}/{file_name}'
    get_logger().info(f'Downloading {url} to {path}')
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        f.write(resp.content)
    if _md5(tmp_path) != md5:
        os.remove(tmp_path)
        raise RuntimeError(f'MD5 mismatch for {url}')
    os.replace(tmp_path, path)
    return path


def _strip_labels(options):
    """Remove the ``A.``-style label of each option.

    The original files mix ``A.``, ``A．``, ``A,``, ``A:`` and ``A `` labels,
    and a few options have no label. A label is only removed when its letter
    matches the option position. A bare space after the letter counts as a
    label only if all four options use it, so options such as ``A man ...``
    are kept intact.
    """
    sep_pat = [
        re.compile(rf'^{letter}\s*[{_SEP}]\s*', re.I) for letter in 'ABCD'
    ]
    space_pat = [re.compile(rf'^{letter}\s+') for letter in 'ABCD']
    use_space = all(s.match(o) for s, o in zip(space_pat, options))
    stripped = []
    for p, s, o in zip(sep_pat, space_pat, options):
        m = p.match(o) or (use_space and s.match(o))
        stripped.append(o[m.end():] if m else o)
    return stripped


def _parse(path):
    """Parse one LogiQA txt file.

    Each example takes 8 lines: blank, answer (a-d), context, query and four
    options.
    """
    with open(path, encoding='utf-8') as f:
        lines = [line.strip() for line in f.read().split('\n')]
    n = len(lines) // 8
    if any(lines[8 * n:]):
        raise ValueError(f'Unexpected trailing lines in {path}')
    data = []
    for k in range(n):
        block = lines[8 * k:8 * k + 8]
        if block[0] or block[1] not in ('a', 'b', 'c', 'd'):
            raise ValueError(f'Malformed example {k} in {path}')
        a, b, c, d = _strip_labels(block[4:8])
        data.append({
            'context': block[2],
            'query': block[3],
            'A': a,
            'B': b,
            'C': c,
            'D': d,
            'correct_option': block[1].upper(),
        })
    return data


@LOAD_DATASET.register_module()
class LogiQADataset(BaseDataset):

    @staticmethod
    def load(path: str = _DEFAULT_ROOT, name: str = None, **kwargs):
        """Load LogiQA from the original release files.

        Args:
            path: Directory holding the original txt files. Missing files are
                downloaded from the pinned upstream commit.
            name: ``'en'`` or ``'zh'``.
        """
        if path in _LEGACY_IDS:
            name = name or _LEGACY_IDS[path]
            path = _DEFAULT_ROOT
        if name not in _FILES:
            raise ValueError(f"name must be 'en' or 'zh', got {name!r}")
        if not osp.isabs(path):
            path = osp.join(os.environ.get('COMPASS_DATA_CACHE', ''), path)
        dataset = DatasetDict()
        for split, (file_name, md5) in _FILES[name].items():
            file_path = _fetch(path, file_name, md5)
            dataset[split] = Dataset.from_list(_parse(file_path))
        return dataset
