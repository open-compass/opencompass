from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    begin='',
    round=[
        dict(role='HUMAN', begin='Question: ', end='\n\n'),
        dict(role='BOT', begin='Answer: ', end='\n\n', generate=True),
    ],
)

models = [
    dict(
        abbr='arithmo-mistral-7b-hf',
        type=HuggingFaceCausalLM,
        path='akjindal53244/Arithmo-Mistral-7B',
        tokenizer_path='akjindal53244/Arithmo-Mistral-7B',
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
