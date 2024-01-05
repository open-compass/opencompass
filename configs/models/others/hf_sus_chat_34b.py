from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='### Human: ', end='\n\n'),
        dict(role="BOT", begin='### Assistant: ', end='\n\n', generate=True),
    ],
    eos_token_id=2
)

models = [
    dict(
        abbr='sus-chat-34b-hf',
        type=HuggingFaceCausalLM,
        path='SUSTech/SUS-Chat-34B',
        tokenizer_path='SUSTech/SUS-Chat-34B',
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
    )
]
