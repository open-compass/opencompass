from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    begin='Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n',
    round=[
        dict(role='HUMAN', begin='### Instruction:\n', end='\n\n'),
        dict(role='BOT', begin='### Response:', end='\n\n', generate=True),
    ],
)

models = [
    dict(
        abbr='gsm8k-rft-llama7b2-u13b',
        type=HuggingFaceCausalLM,
        path='OFA-Sys/gsm8k-rft-llama7b2-u13b',
        tokenizer_path='OFA-Sys/gsm8k-rft-llama7b2-u13b',
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
