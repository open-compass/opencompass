# This config is used for pass@k evaluation with dataset repetition
# That model cannot generate multiple response for single input
from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner
from opencompass.models import HuggingFaceCausalLM
from opencompass.runners import LocalRunner
from opencompass.partitioners import SizePartitioner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .datasets.humaneval.humaneval_repeat10_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.deprecated_mbpp_repeat10_gen_1e1056 import mbpp_datasets
    from .datasets.mbpp.deprecated_sanitized_mbpp_repeat10_gen_1e1056 import sanitized_mbpp_datasets

datasets = []
datasets += humaneval_datasets
datasets += mbpp_datasets
datasets += sanitized_mbpp_datasets

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
)

models = [
    dict(
        abbr='internlm-chat-7b-hf-v11',
        type=HuggingFaceCausalLM,
        path='internlm/internlm-chat-7b-v1_1',
        tokenizer_path='internlm/internlm-chat-7b-v1_1',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_seq_len=2048,
        meta_template=_meta_template,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        generation_kwargs=dict(
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        ),
        run_cfg=dict(num_gpus=1, num_procs=1),
        batch_size=8,
    )
]


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=600),
    runner=dict(
        type=LocalRunner, max_num_workers=16,
        task=dict(type=OpenICLInferTask)),
)
