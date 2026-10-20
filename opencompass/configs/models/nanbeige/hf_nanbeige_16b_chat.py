from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='### Human: \n', end='\n\n'),
        dict(role='BOT', begin='### Assistant: ', end='</s>', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='nanbeige-16b-chat-hf',
        path='Nanbeige/Nanbeige-16B-Chat',
        tokenizer_path='Nanbeige/Nanbeige-16B-Chat',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            torch_dtype='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=_meta_template,
        batch_padding=False,
        max_out_len=100,
        max_seq_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=2, num_procs=1),
        end_str='</s>',
    )
]
