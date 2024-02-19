from opencompass.models import ModelScopeCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role='BOT', begin='\n<|im_start|>assistant\n', end='<|im_end|>', generate=True),
    ],
)

models = [
    dict(
        type=ModelScopeCausalLM,
        abbr='qwen-7b-chat-ms',
        path='qwen/Qwen-7B-Chat',
        tokenizer_path='qwen/Qwen-7B-Chat',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,),
        pad_token_id=151643,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
