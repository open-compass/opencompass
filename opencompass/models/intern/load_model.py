import io
import json
import os
import os.path as osp
from typing import Optional

import internlm
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from sentencepiece import SentencePieceProcessor

from .generation_tools import LLMGenerator, LLMTokenizer
from .utils.checkpoint_utils import merge_pp_within_tp
from .utils.dict import (basic_config, convert2run, gen_masks,
                         local_config_convert, maxdim2oridim, maxlay2orilay,
                         model_name2iter_info, model_name2tokenizer_info)
from .utils.utils import proxy_off, try_import_petrel_client

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

    # init colossalai paralleling config
    WORLD_SIZE = os.getenv('WORLD_SIZE')
    if WORLD_SIZE is None:
        print('Supposed to launch with torchrun!')
        exit()
    # TP = int(WORLD_SIZE)
    ckpts = checkpoint.split(';')
    assert ckpts
    ckpt = str(ckpts[0])
    # parameter splitting:
    if model_config_path is not None:
        model_name = osp.realpath(ckpt).split('/')[-1]
        cur_iter = None
    elif 's3://' in ckpt:
        # We assume that the path looks like the following:
        # opennlplab_hdd:s3://opennlplab_hdd/llm_it/0419/sft_7132k_flan64_8196/1399
        model_name, cur_iter = osp.realpath(ckpt).split('/')[-2:]
        try:
            cur_iter = int(cur_iter)
        except:  # allow s3 path without iteration.
            cur_iter = 0
        assert client.isdir(ckpt)
    else:
        model_name, cur_iter = osp.realpath(ckpt).split('/')[-2:]
        try:
            cur_iter = int(cur_iter)
        except:
            if ckpt.startswith('/cpfs01/'):
                cur_iter = 0
            else:
                print('Something mistakes')
                exit(-1)

        # 'model_tp0_pp*.pt' ~ 'model_tp7_pp*.pt'
        save_tp = 0
        for file in os.listdir(ckpt):
            if file.startswith('model_tp') and file.endswith('.pt'):
                save_tp = max(save_tp, int(file[8:].split('_')[0]))
        save_tp += 1

    # internlm.launch_from_torch(
    #     config={
    #         "parallel": dict(
    #             zero1=1,
    #         )
    #     }, seed=42)
    print(model_config_path)
    internlm.launch_from_torch(config=model_config_path, seed=42)

    # print args info
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    if tp_rank == 0:
        print(f'Args: ckpt={checkpoint}')

    tokenizer = SentencePieceProcessor()
    if not tokenizer_path:
        # TODO: delete these hardcoded mapping
        tokenizer_path, tokenizer_type = model_name2tokenizer_info(model_name)
        if 'llamav4.model' in tokenizer_path:
            tokenizer_path = \
                '/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model'
    tokenizer.load(tokenizer_path)
    tokenizer = LLMTokenizer(tokenizer,
                             max_seq_len=max_seq_len,
                             tokenizer_type=tokenizer_type)

    if model_config_path is not None:
        print('Beginning to load model_config', flush=True)
        update_config = gpc.config.model
        print('Config done!', flush=True)
    elif cur_iter is None:
        print('Beginning to load model_config', flush=True)
        assert ckpt.startswith('/mnt/petrelfs/share_data/llm_llama/')
        config_path = os.path.join(ckpt, 'params.json')
        with open(config_path, 'r') as f:
            update_config = local_config_convert(json.load(f))
        assert update_config['vocab_size'] == -1, '?'
        update_config['vocab_size'] = tokenizer.vocab_size()
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

    model_config = convert2run(model_config)
    model_module = module
    if tokenizer_type in ['llama', 'v6', 'v4']:
        model_config['embed_split_hidden'] = True

    if 'layer_norm_epsilon' in model_config:
        del model_config['layer_norm_epsilon']

    model = model_module(**model_config)
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

    warm_iter, max_iter = model_name2iter_info(model_name)
    use_mask = (warm_iter + max_iter) != 0

    if use_mask:
        feat_mask, ffn_mask, layer_mask = gen_masks(
            cur_iter=cur_iter,
            maxdim=model_config['hidden_size'],
            oridim=maxdim2oridim[model_config['hidden_size']],
            maxlay=model_config['num_layers'],
            orilay=maxlay2orilay[model_config['num_layers']],
            warm_iter=warm_iter,
            max_iter=max_iter)
    else:
        feat_mask, ffn_mask, layer_mask = None, None, None

    torch.distributed.barrier()

    generator = LLMGenerator(model,
                             tokenizer,
                             use_mask,
                             forward_kwargs={
                                 'feat_mask': feat_mask,
                                 'ffn_mask': ffn_mask,
                                 'layer_mask': layer_mask
                             })

    return model, tokenizer, generator, tp_rank
