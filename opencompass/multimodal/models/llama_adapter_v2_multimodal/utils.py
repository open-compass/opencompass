import hashlib
import os
import urllib
import warnings

import torch
from tqdm import tqdm


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def format_prompt(instruction, input=None):

    PROMPT_DICT = {
        'prompt_input':
        ('Below is an instruction that describes a task, paired with an input that provides further context. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
         ),
        'prompt_no_input':
        ('Below is an instruction that describes a task. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction}\n\n### Response:'),
    }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map(
            {'instruction': instruction})
    else:
        return PROMPT_DICT['prompt_input'].format_map({
            'instruction': instruction,
            'input': input
        })


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    # assume the url is https://some/path/sha256_model.pth
    expected_sha256 = url.split('/')[-1].split('_')[0]
    # expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f'{download_target} exists and is not a regular file')

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target,
                               'rb').read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f'{download_target} exists, but the SHA256 checksum does not match; re-downloading the file'
            )

    with urllib.request.urlopen(url) as source, open(download_target,
                                                     'wb') as output:
        with tqdm(total=int(source.info().get('Content-Length')),
                  ncols=80,
                  unit='iB',
                  unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target,
                           'rb').read()).hexdigest() != expected_sha256:
        raise RuntimeError(
            'Model has been downloaded but the SHA256 checksum does not not match'
        )

    return download_target
