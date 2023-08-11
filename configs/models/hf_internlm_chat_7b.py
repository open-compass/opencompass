from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='<eoh>\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm-chat-7b-hf',
        path="internlm/internlm-chat-7b",
        tokenizer_path='internlm/internlm-chat-7b',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
            revision="1a6328795c6e207904e1eb58177e03ad24ae06f3"
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template,
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            revision="1a6328795c6e207904e1eb58177e03ad24ae06f3"),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
