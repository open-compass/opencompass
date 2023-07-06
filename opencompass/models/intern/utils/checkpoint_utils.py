import io
import os
import re

import torch
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc

from .storage_manager import get_storage_manager
from .utils import try_import_petrel_client

Client = try_import_petrel_client()


def _check_folder(folder):
    get_storage_manager().check_folder(folder)


def _get_fns(folder):
    return get_storage_manager().get_fns(folder)


def load_with_progress_bar(fp, disable=True):
    # size = get_s3_object_file_size(fp)
    # pbar = tqdm(total=size, leave=False, disable=disable)
    client = Client()
    stream = client.get(fp, enable_stream=True)
    # chunk_size = min(size, 8192)
    f = io.BytesIO()
    for chunk in stream.iter_chunks(chunk_size=8192):
        f.write(chunk)
        # pbar.update(chunk_size)
    f.seek(0)
    return f


def _auto_load_with_bar(fp, disable=True):
    if 's3://' in fp:
        client = Client()
        with load_with_progress_bar(fp, disable=disable) as f:
            states = torch.load(f, map_location='cpu')
    else:
        states = torch.load(fp, map_location='cpu')
    return states


def merge_pp_within_tp(folder, local_rank=None):
    """给定一个 folder ，merge 下面的 pipeline model."""
    _check_folder(folder)
    fns = _get_fns(folder)

    model_fns = []
    for fn in fns:
        if fn.startswith('model_t') and not fn.endswith(
                'md5'):  # 加入 _t 是为了避免和model_config.py冲突
            model_fns.append(fn)

    max_tp, max_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split('_')
        max_pp = max(max_pp, int(pp[2:]) + 1)
        max_tp = max(max_tp, int(tp[2:]) + 1)

    if local_rank is None:
        assert max_tp == gpc.get_world_size(
            ParallelMode.TENSOR
        ), f'The model trained with tp:{max_tp}, but current tp:{gpc.get_world_size(ParallelMode.TENSOR)}'
        tp = gpc.get_local_rank(ParallelMode.TENSOR)
    else:
        tp = local_rank

    layer_shift = 0

    tp_states = {}
    for pp in range(max_pp):
        _layer_shift = 0
        model_name = f'model_tp{tp}_pp{pp}.pt'
        states = _auto_load_with_bar(os.path.join(folder, model_name),
                                     disable=tp != 0)
        keys = list(states.keys())
        for key in keys:
            match = re.search('\.\d+\.', key)
            if match is not None:  # 说明是 layer 相关的, 需要shift
                s, e = match.span()
                layer_idx = int(key[s + 1:e - 1]) + layer_shift
                _layer_shift = max(_layer_shift, int(key[s + 1:e - 1]))
                name = key[:s] + f'.{layer_idx}.' + key[e:]
                tp_states[name] = states[key]
            else:
                tp_states[key] = states[key]
        layer_shift += _layer_shift + 1

    return {(key[6:] if key.startswith('model.') else key): value
            for key, value in tp_states.items()}
