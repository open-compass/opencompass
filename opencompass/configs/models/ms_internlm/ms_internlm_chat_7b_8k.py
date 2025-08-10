from opencompass.models import ModelScopeCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='<eoh>\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
)

models = [
    dict(
        type=ModelScopeCausalLM,
        abbr='internlm-chat-7b-8k-ms',
        path='Shanghai_AI_Laboratory/internlm-chat-7b-8k',
        tokenizer_path='Shanghai_AI_Laboratory/internlm-chat-7b-8k',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
