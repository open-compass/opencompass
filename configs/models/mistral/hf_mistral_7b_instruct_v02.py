from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    begin="<s>",
    round=[
        dict(role="HUMAN", begin='[INST]', end='[/INST]'),
        dict(role="BOT", begin="", end='</s>', generate=True),
    ],
    eos_token_id=2
)

models = [
    dict(
        abbr='mistral-7b-instruct-v0.2-hf',
        type=HuggingFaceCausalLM,
        path='mistralai/Mistral-7B-Instruct-v0.2',
        tokenizer_path='mistralai/Mistral-7B-Instruct-v0.2',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='</s>',
    )
]


models_sample = [
    dict(
        abbr='mistral-7b-instruct-v0.2-hf',
        type=HuggingFaceCausalLM,
        path='mistralai/Mistral-7B-Instruct-v0.2',
        tokenizer_path='mistralai/Mistral-7B-Instruct-v0.2',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
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
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='</s>',
    )
]
