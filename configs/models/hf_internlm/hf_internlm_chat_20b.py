from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm-chat-20b-hf',
        path="internlm/internlm-chat-20b",
        tokenizer_path='internlm/internlm-chat-20b',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=2, num_procs=1),
        end_str='<eoa>',
    )
]

models_sample = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm-chat-20b-hf',
        path="internlm/internlm-chat-20b",
        tokenizer_path='internlm/internlm-chat-20b',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        generation_kwargs=dict(
            do_sample= True,
            temperature = 1.0,
            top_p = 0.8,
            top_k = 0,
            repetition_penalty = 1.1
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=2, num_procs=1),
        end_str='<eoa>',
    )
]
