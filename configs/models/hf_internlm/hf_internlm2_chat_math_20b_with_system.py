from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='SYSTEM', begin='<|im_start|>system\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>\n', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm2-chat-math-20b-hf',
        path='internlm/internlm2-math-20b',
        tokenizer_path='internlm/internlm2-math-20b',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
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
        run_cfg=dict(num_gpus=2, num_procs=1),
        end_str='<|im_end|>',
    )
]
