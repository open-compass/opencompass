from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.models import VLLM_API

with read_base():
    from ..summarizers.medium import summarizer
    from ..datasets.mmlu.mmlu_ppl import mmlu_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = [
    dict(
        type= VLLM_API,
        abbr='Qwen-7b-api',
        path = '',
        url="http://0.0.0.0:60/generate",
        max_seq_len=2048,
        batch_size=1000,
        generation_kwargs = {
            'temperature': 0.8,
            'max_out_len': 1,
            'prompt_logprobs': 0,

        },),
]

