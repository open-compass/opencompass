from opencompass.models import LMDeploywithChatTemplate

# inference backend of LMDeploy. It can be one of ['turbomind', 'pytorch']
backend = 'turbomind'
# the config of the inference backend
# For the detailed config, please refer to 
# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/messages.py
engine_config = dict(
    backend=backend,
    turbomind=dict(
        max_batch_size=128,
        tp=1,
        quant_policy='0',
        model_format='hf',
        enable_prefix_caching=False,
    ),
    pytorch=dict(
        max_batch_size=128,
        tp=1,
        enable_prefix_caching=False
    )
)

models = [
    dict(
        type=LMDeploywithChatTemplate,
        abbr=f'internlm2-chat-7b-{backend}',
        path='internlm/internlm2-chat-7b',
        engine_config=engine_config,
        gen_config=dict(
            top_k=1,
        ),
        max_seq_len=8000,
        max_out_len=1024,
        # the max number of prompts that LMDeploy receives
        # in `generate` function 
        batch_size=5000,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
    )
]
