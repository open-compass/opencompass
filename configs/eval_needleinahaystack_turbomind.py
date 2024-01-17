from opencompass.models.turbomind import TurboMindModel

from mmengine.config import read_base
with read_base():
    from .datasets.cdme.cdme200k import cdme_datasets

datasets = [*cdme_datasets]

internlm_meta_template = dict(round=[
    dict(role='HUMAN', begin='<|User|>:', end='\n'),
    dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
],
                              eos_token_id=103028)

models = [
    # config for internlm-chat-20b
    dict(
        type=TurboMindModel,
        abbr='internlm-chat-20b-turbomind',
        path='./turbomind',
        max_out_len=100,
        max_seq_len=201000,
        batch_size=8,
        concurrency=8,
        meta_template=internlm_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
