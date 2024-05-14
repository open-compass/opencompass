# This config is used for pass@k evaluation with `num_return_sequences`
# That model can generate multiple responses for single input
from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner
from opencompass.models import HuggingFaceCausalLM
from opencompass.runners import LocalRunner
from opencompass.partitioners import SizePartitioner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .datasets.humaneval.humaneval_passk_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.deprecated_mbpp_passk_gen_1e1056 import mbpp_datasets
    from .datasets.mbpp.deprecated_sanitized_mbpp_passk_gen_1e1056 import sanitized_mbpp_datasets

datasets = []
datasets += humaneval_datasets
datasets += mbpp_datasets
datasets += sanitized_mbpp_datasets

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='CodeLlama-7b-Python',
        path='codellama/CodeLlama-7b-Python-hf',
        tokenizer_path='codellama/CodeLlama-7b-Python-hf',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        generation_kwargs=dict(
            num_return_sequences=10,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        ),
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=300),
    runner=dict(
        type=LocalRunner, max_num_workers=16,
        task=dict(type=OpenICLInferTask)),
)
