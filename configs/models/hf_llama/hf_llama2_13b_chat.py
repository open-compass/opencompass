from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='[INST] ', end=' '),
        dict(role="BOT", begin='[/INST] ', end=' ', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-2-13b-chat-hf',
        path="meta-llama/Llama-2-13b-chat-hf",
        tokenizer_path='meta-llama/Llama-2-13b-chat-hf',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        batch_padding=False,
        run_cfg=dict(num_gpus=2, num_procs=1),
        end_str='[INST]',
    )
]

models_sample = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-2-13b-chat-hf',
        path="meta-llama/Llama-2-13b-chat-hf",
        tokenizer_path='meta-llama/Llama-2-13b-chat-hf',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
        ),
        generation_kwargs=dict(
            do_sample= True,
            temperature = 0.6,
            top_p = 0.9,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        batch_padding=False,
        run_cfg=dict(num_gpus=2, num_procs=1),
        end_str='[INST]',
    )
]