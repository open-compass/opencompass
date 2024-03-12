from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='### Instruction:\n', end='\n'),
        dict(role="BOT", begin="### Response:\n", end='<|EOT|>', generate=True),
    ],
    eos_token_id=100001,
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='deepseek-coder-6.7b-hf',
        path="deepseek-ai/deepseek-coder-6.7b-instruct",
        tokenizer_path='deepseek-ai/deepseek-coder-6.7b-instruct',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=_meta_template,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<|EOT|>',
    )
]