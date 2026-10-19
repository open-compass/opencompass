from opencompass.models import TurboMindModelwithChatTemplate


models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr=f'internlm2-chat-7b-lmdeploy',
        path='internlm/internlm2-chat-7b',
        # inference backend of LMDeploy. It can be either 'turbomind' or 'pytorch'.
        # If the model is not supported by 'turbomind', it will fallback to
        # 'pytorch'
        backend='turbomind',
        # For the detailed engine config and generation config, please refer to
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/messages.py
        engine_config=dict(tp=1),
        gen_config=dict(do_sample=False),
        max_seq_len=8192,
        max_out_len=4096,
        # the max number of prompts that LMDeploy receives
        # in `generate` function
        batch_size=5000,
        run_cfg=dict(num_gpus=1),
    )
]
