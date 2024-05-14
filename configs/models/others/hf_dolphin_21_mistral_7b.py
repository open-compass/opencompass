from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>\n', generate=True),
    ],
)

models = [
    dict(
        abbr='dolphin-2.2.1-mistral-7b-hf',
        type=HuggingFaceCausalLM,
        path='ehartford/dolphin-2.2.1-mistral-7b',
        tokenizer_path='ehartford/dolphin-2.2.1-mistral-7b',
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
    )
]
