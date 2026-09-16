from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='Weddoo-Doo-Q3.8.W9',
        path='Weddoo/Weddoo-Doo-Q3.8.W9',
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
        max_seq_len=8192,
        max_out_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
