import io
import json
import os
from typing import Optional

import internlm
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from sentencepiece import SentencePieceProcessor

from .utils.checkpoint_utils import merge_pp_within_tp
from .utils.generation_tools import LLMGenerator, LLMTokenizer
from .utils.utils import (basic_config, convert2run, proxy_off,
                          try_import_petrel_client)

Client = try_import_petrel_client()


def setup_model_parallel(init_seed=1):
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))

    torch.distributed.init_process_group('nccl')
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(init_seed)
    return local_rank, world_size


def load_llm(checkpoint,
             max_seq_len=2048,
             tokenizer_path: Optional[str] = None,
             tokenizer_type: Optional[str] = None,
             module=None,
             model_config_path=None):
    proxy_off()
    client = Client()
    WORLD_SIZE = os.getenv('WORLD_SIZE')
    if WORLD_SIZE is None:
        print('Supposed to launch with torchrun!')
        exit()
    ckpts = checkpoint.split(';')
    assert ckpts
    ckpt = str(ckpts[0])
    # parameter splitting:
    if model_config_path is None:
        internlm.launch_from_torch(config={'parallel': dict(zero1=1, )},
                                   seed=42)
    else:
        internlm.launch_from_torch(config=model_config_path, seed=42)

    # print args info
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    if tp_rank == 0:
        print(f'Args: ckpt={checkpoint}')

    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    tokenizer = LLMTokenizer(tokenizer,
                             max_seq_len=max_seq_len,
                             tokenizer_type=tokenizer_type)

    if model_config_path is not None:
        print('Beginning to load model_config', flush=True)
        update_config = gpc.config.model
        print('Config done!', flush=True)
    else:
        print('Config loading', flush=True)
        config_file = os.path.join(ckpt, 'model_config.pt')
        if 's3://' in ckpt:
            if client.contains(config_file):
                with io.BytesIO(client.get(config_file)) as f:
                    update_config = torch.load(f)
            else:
                config_file = os.path.join(ckpt, 'params.json')
                update_config = json.loads(client.get(config_file).decode())
        else:
            with open(config_file, 'rb') as f:
                update_config = torch.load(f)
        print('Config done!', flush=True)

    model_config = basic_config
    model_config.update(update_config)
    model_config = convert2run(model_config, tokenizer_type)
    model = module(**model_config)
    states = merge_pp_within_tp(ckpt, tp_rank)
    if len(ckpts) > 1:
        for ckpt_ in ckpts[1:]:
            states_ = merge_pp_within_tp(ckpt_)
            for k in states_.keys():
                states[k] += states_[k]

        for k in states.keys():
            states[k] /= len(ckpts)

    load_info = model.load_state_dict(states, strict=False)
    if tp_rank == 0:
        print(load_info)
        if load_info.missing_keys:
            exit(-1)
    model = model.half().eval().cuda()

    torch.distributed.barrier()
    use_mask = False
    generator = LLMGenerator(model,
                             tokenizer,
                             use_mask,
                             forward_kwargs={
                                 'feat_mask': None,
                                 'ffn_mask': None,
                                 'layer_mask': None
                             })

    return model, tokenizer, generator, tp_rank
