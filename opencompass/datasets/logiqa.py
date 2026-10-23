import re

from datasets import (Dataset, DatasetDict, DownloadConfig, DownloadManager,
                      load_dataset)

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

# The HF mirrors below only host a loading script that downloads the original
# txt files from https://github.com/lgw863/LogiQA-dataset. Their
# `_process_answer` strips option labels with a fixed `answer[3:]`, which drops
# the first character of options written as `A.20...` or `A．一...`. We
# reproduce the scripts here with the label stripping fixed.
_URL = 'https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master'
_HF_SCRIPTS = {
    'lucasmccabe/logiqa': {
        'train': 'Train.txt',
        'validation': 'Eval.txt',
        'test': 'Test.txt',
    },
    'jiacheng-ye/logiqa-zh': {
        'train': 'zh_train.txt',
        'validation': 'zh_eval.txt',
        'test': 'zh_test.txt',
    },
}


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
    def load(path: str, **kwargs):
        if path in _HF_SCRIPTS:
            # Same URLs and download manager as the HF scripts; files are
            # cached under the HF datasets cache.
            # use_etag=False lets later runs (incl. offline) hit the cache
            # without a HEAD request.
            dl_manager = DownloadManager(download_config=DownloadConfig(
                use_etag=False))
            files = dl_manager.download_and_extract({
                split: f'{_URL}/{name}'
                for split, name in _HF_SCRIPTS[path].items()
            })
            return DatasetDict({
                split: Dataset.from_list(_parse(file))
                for split, file in files.items()
            })

        dataset = load_dataset(path=path, **kwargs)

        def preprocess(example):
            options = example.pop('options')
            example['A'] = options[0]
            example['B'] = options[1]
            example['C'] = options[2]
            example['D'] = options[3]
            example['correct_option'] = chr(65 + example['correct_option'])
            return example

        dataset = dataset.map(preprocess)
        return dataset
