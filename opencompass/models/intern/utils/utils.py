import io
import os
import re

import torch
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc

from .storage_manager import get_storage_manager

basic_config = dict(num_chunks=1,
                    checkpoint=False,
                    dtype=torch.half,
                    embed_split_hidden=False,
                    num_layers=40,
                    hidden_size=5120,
                    vocab_size=150494,
                    embed_grad_scale=1,
                    parallel_output=False,
                    num_attention_heads=40,
                    mlp_ratio=8 / 3,
                    apply_post_layer_norm=False,
                    residual_in_fp32=False,
                    norm_type='rmsnorm',
                    drop_rate=0,
                    attn_drop_rate=0)

backup = {}


def try_import_petrel_client():
    """
    Overview:
        Try import petrel_client module, if failed, return ``None``

    Returns:
        - (:obj:`Module`): Imported module,
            or ``None`` when petrel_client not found
    """
    try:
        from petrel_client.client import Client

        return Client
    except ModuleNotFoundError as e:
        print(f'petrel_client.client import error! {e}', flush=True)
        return lambda *args, **kwargs: None


Client = try_import_petrel_client()


def _check_folder(folder):
    get_storage_manager().check_folder(folder)


def _get_fns(folder):
    return get_storage_manager().get_fns(folder)


def load_with_progress_bar(fp, disable=True):
    client = Client()
    stream = client.get(fp, enable_stream=True)
    f = io.BytesIO()
    for chunk in stream.iter_chunks(chunk_size=8192):
        f.write(chunk)
    f.seek(0)
    return f


def _auto_load_with_bar(fp, disable=True):
    states = torch.load(fp, map_location='cpu')
    return states


def merge_pp_within_tp(folder, local_rank=None):
    _check_folder(folder)
    fns = _get_fns(folder)

    model_fns = []
    for fn in fns:
        if fn.startswith('model_t') and not fn.endswith('md5'):
            model_fns.append(fn)

    max_tp, max_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split('_')
        max_pp = max(max_pp, int(pp[2:]) + 1)
        max_tp = max(max_tp, int(tp[2:]) + 1)

    if local_rank is None:
        assert max_tp == gpc.get_world_size(
            ParallelMode.TENSOR
        ), f'The model trained with tp:{max_tp}, but current tp:{gpc.get_world_size(ParallelMode.TENSOR)}'  # noqa: E501
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
            match = re.search('\.\d+\.', key)  # noqa: W605
            if match is not None:
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


def proxy_off():
    global backup
    if 'http_proxy' in os.environ:
        backup['http_proxy'] = os.environ['http_proxy']
        del os.environ['http_proxy']
    if 'https_proxy' in os.environ:
        backup['https_proxy'] = os.environ['https_proxy']
        del os.environ['https_proxy']
    if 'HTTP_PROXY' in os.environ:
        backup['HTTP_PROXY'] = os.environ['HTTP_PROXY']
        del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ:
        backup['HTTPS_PROXY'] = os.environ['HTTPS_PROXY']
        del os.environ['HTTPS_PROXY']


def proxy_on():
    global backup
    if 'http_proxy' in backup:
        os.environ['http_proxy'] = backup['http_proxy']
    if 'https_proxy' in backup:
        os.environ['https_proxy'] = backup['https_proxy']
    if 'HTTP_PROXY' in backup:
        os.environ['HTTP_PROXY'] = backup['HTTP_PROXY']
    if 'HTTPS_PROXY' in backup:
        os.environ['HTTPS_PROXY'] = backup['HTTPS_PROXY']


def convert2run(model_config, tokenizer_type):
    model_config['dtype'] = torch.half if str(
        model_config['dtype']) == 'torch.float16' else torch.bfloat16
    model_config['parallel_output'] = False
    model_config.pop('no_bias', None)
    model_config.pop('deepnorm', None)
    model_config.pop('model_type', None)
    if tokenizer_type in ['llama', 'v6', 'v4']:
        model_config['embed_split_hidden'] = True
    if 'layer_norm_epsilon' in model_config:
        del model_config['layer_norm_epsilon']
    return model_config
