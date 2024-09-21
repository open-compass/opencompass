from opencompass.models import LMDeploywithChatTemplate


models = [
    dict(
        type=LMDeploywithChatTemplate,
        abbr=f'internlm2-chat-7b-lmdeploy',
        path='internlm/internlm2-chat-7b',
        # inference backend of LMDeploy. It can be one of ['turbomind', 'pytorch']
        backend='turbomind',
        # For the detailed config, please refer to
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/messages.py
        engine_config=dict(
            dtype='auto',
            max_batch_size=256,
            tp=1,
            quant_policy=0,
            enable_prefix_caching=False,
        ),
        gen_config=dict(
            do_sample=False
        ),
        max_seq_len=8000,
        # the max number of prompts that LMDeploy receives
        # in `generate` function
        batch_size=5000,
        run_cfg=dict(num_gpus=1),
    )
]
