from mmengine.config import read_base
from opencompass.models.turbomind import TurboMindModel

with read_base():
    # choose a list of datasets
    from .datasets.SuperGLUE_CB.SuperGLUE_CB_gen import CB_datasets
    # and output the results in a choosen format
    from .summarizers.medium import summarizer

datasets = [*CB_datasets]


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='<eoh>\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
)

models = [
    dict(
        type=TurboMindModel,
        abbr='internlm-chat-7b-tb',
        path="internlm-chat-7b",
        model_path='./workspace',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        meta_template=_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
