from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='GPT4 Correct User: ', end='<|end_of_turn|>'),
        dict(role='BOT', begin='GPT4 Correct Assistant: ', end='<|end_of_turn|>', generate=True),
    ],
)

models = [
    dict(
        abbr='openchat-3.5-0106-hf',
        type=HuggingFaceCausalLM,
        path='openchat/openchat-3.5-0106',
        tokenizer_path='openchat/openchat-3.5-0106',
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
        end_str='<|end_of_turn|>',
    )
]
