from mmengine.config import read_base
from opencompass.models.turbomind import TurboMindModel

with read_base():
    # choose a list of datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # and output the results in a choosen format
    from .summarizers.medium import summarizer

datasets = [*gsm8k_datasets]


meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
    eos_token_id=103028)

models = [
    dict(
        type=TurboMindModel,
        abbr='llama2-chat-7b-turbomind',
        path="internlm-chat-7b",
        tis_addr='0.0.0.0:63337',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        meta_template=meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
