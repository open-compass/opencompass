from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM


with read_base():
    from .datasets.winogrande.winogrande_gen_a027b6 import winogrande_datasets

datasets = [*winogrande_datasets]

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='<eoh>\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
)

models=[
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm-chat-7b-hf',
        path='internlm/internlm-chat-7b',
        tokenizer_path='internlm/internlm-chat-7b',
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
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

_winogrande_all = [d['abbr'] for d in winogrande_datasets]

summarizer = dict(
    summary_groups=[
        {'name': 'winogrande', 'subsets': _winogrande_all},
        {'name': 'winogrande_std', 'subsets': _winogrande_all, 'std': True},
    ]
)
