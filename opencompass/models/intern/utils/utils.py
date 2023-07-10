import os

import torch

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
