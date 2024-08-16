from opencompass.partitioners import (
    NaivePartitioner,
    NumWorkerPartitioner,
)
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from ..configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat_1m import (
        models as internlm2_5_7b_chat_1m,
    )
    from .datasets.ruler.ruler_32k_gen import ruler_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
models = internlm2_5_7b_chat_1m
work_dir = './outputs/ruler'


infer = dict(
    partitioner=dict(type=NumWorkerPartitioner),
    runner=dict(
        type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLInferTask), retry=5
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=32, task=dict(type=OpenICLEvalTask)),
)

abbr_suffixs = ['4k', '8k', '16k', '32k']
summary_groups = []
for abbr_suffix in abbr_suffixs:
    summary_groups.append(
        {
            'name': f'ruler_{abbr_suffix}',
            'subsets': [
                dataset['abbr']
                for dataset in datasets
                if abbr_suffix in dataset['abbr']
            ],
        }
    )
summarizer = dict(
    dataset_abbrs=[f'ruler_{abbr_suffix}' for abbr_suffix in abbr_suffixs], summary_groups=summary_groups
)
