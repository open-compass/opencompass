import os.path as osp
import re

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

# Original release: https://github.com/lgw863/LogiQA-dataset


def _process_sentences(text):
    # Copied from the HF loading scripts to keep the text unchanged.
    text = text.replace('\n', '')
    sents = text.split('.')
    text = ''
    for sent in sents:
        if len(sent) == 0:
            continue
        if len(text) == 0:
            text += sent
        elif sent[0].isnumeric():
            text += '.' + sent
        else:
            text += '. ' + sent
    text = text.replace('  ', ' ')
    text = text.replace("\\'", "'")
    while text.endswith(' '):
        text = text[:-1]
    if re.match(r'^[A-Z][\w\s]+[?.!]$', text) is None:
        text += '.'
    text = text.replace('?.', '?')
    text = text.replace('!.', '!')
    text = text.replace('..', '.')
    return text


def _strip_labels(options):
    """Remove the ``A. ``-style label of each option.

    Labels mix ``.``, ``．``, ``,``, ``:`` and bare spaces. A bare space after
    the letter counts as a label only if all four options use it, so options
    such as ``A man ...`` are kept intact.
    """
    sep_pat = re.compile(r'^[A-D]\s*[.．,，:：、。]\s*')
    space_pat = re.compile(r'^[A-D]\s+')
    use_space = all(space_pat.match(o) for o in options)
    stripped = []
    for o in options:
        m = sep_pat.match(o) or (use_space and space_pat.match(o))
        # Drop at most 3 chars like the original `answer[3:]`, so only the
        # options it truncated change.
        stripped.append(o[min(3, m.end()):] if m else o)
    return stripped


def _parse(path):
    with open(path, encoding='utf-8') as f:
        lines = [_process_sentences(s) for s in f.readlines()]
    data = []
    for key in range(len(lines) // 8):
        row = 8 * key
        options = _strip_labels(lines[row + 4:row + 8])
        data.append({
            'context': lines[row + 2],
            'query': lines[row + 3],
            'A': options[0],
            'B': options[1],
            'C': options[2],
            'D': options[3],
            'correct_option': lines[row + 1].replace('.', '').upper(),
        })
    return data


@LOAD_DATASET.register_module()
class LogiQADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        """Load LogiQA.

        Args:
            path: ``'opencompass/LogiQA'`` or a local directory holding
                the original release txt files.
            name: ``'en'`` or ``'zh'``.
        """
        if name not in ('en', 'zh'):
            raise ValueError(f"name must be 'en' or 'zh', got {name!r}")
        path = get_data_path(path)
        files = {
            'en': ('Train.txt', 'Eval.txt', 'Test.txt'),
            'zh': ('zh_train.txt', 'zh_eval.txt', 'zh_test.txt'),
        }[name]
        return DatasetDict({
            split: Dataset.from_list(_parse(osp.join(path, file_name)))
            for split, file_name in zip(('train', 'validation', 'test'), files)
        })
