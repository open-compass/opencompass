from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    begin='<|startoftext|>',
    round=[
        dict(role='HUMAN', begin='Human: ', end='\n\n'),
        dict(role='BOT', begin='Assistant: <|endoftext|>', end='<|endoftext|>', generate=True),
    ],
)

models = [
    dict(
        abbr='orionstar-yi-34b-chat-hf',
        type=HuggingFaceCausalLM,
        path='OrionStarAI/OrionStar-Yi-34B-Chat',
        tokenizer_path='OrionStarAI/OrionStar-Yi-34B-Chat',
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
        run_cfg=dict(num_gpus=4, num_procs=1),
        end_str='<|endoftext|>',
    )
]
