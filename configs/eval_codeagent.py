from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.models import OpenAI, HuggingFaceCausalLM
from opencompass.models.lagent import CodeAgent

with read_base():
    from .datasets.math.math_gen_943d32 import math_datasets
    from .datasets.gsm8k.gsm8k_gen_57b0b1 import gsm8k_datasets

datasets = []
datasets += gsm8k_datasets
datasets += math_datasets

models = [
    dict(
        abbr='gpt-3.5-react',
        type=CodeAgent,
        llm=dict(
            type=OpenAI,
            path='gpt-3.5-turbo',
            key='ENV',
            query_per_second=1,
            max_seq_len=4096,
        ),
        batch_size=8),
    dict(
        abbr='WizardCoder-Python-13B-V1.0-react',
        type=CodeAgent,
        llm=dict(
            type=HuggingFaceCausalLM,
            path='WizardLM/WizardCoder-Python-13B-V1.0',
            tokenizer_path='WizardLM/WizardCoder-Python-13B-V1.0',
            tokenizer_kwargs=dict(
                padding_side='left',
                truncation_side='left',
                trust_remote_code=True,
            ),
            max_seq_len=2048,
            model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        ),
        batch_size=8,
        run_cfg=dict(num_gpus=2, num_procs=1)),
]

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=40000),
    runner=dict(
        type=LocalRunner, max_num_workers=16,
        task=dict(type=OpenICLInferTask)),
)
