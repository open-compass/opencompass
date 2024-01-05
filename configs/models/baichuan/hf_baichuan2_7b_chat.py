from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<reserved_106>'),
        dict(role='BOT', begin='<reserved_107>', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='baichuan2-7b-chat-hf',
        path="baichuan-inc/Baichuan2-7B-Chat",
        tokenizer_path='baichuan-inc/Baichuan2-7B-Chat',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
models_sample = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='baichuan2-7b-chat-hf',
        path="baichuan-inc/Baichuan2-7B-Chat",
        tokenizer_path='baichuan-inc/Baichuan2-7B-Chat',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        generation_kwargs=dict(
            do_sample= True,
            temperature = 0.3,
            top_p = 0.85,
            top_k = 5,
            repetition_penalty = 1.05
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
