from opencompass.partitioners import (
    NaivePartitioner,
    NumWorkerPartitioner,
)
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat_1m import (
        models as internlm2_5_7b_chat_1m,
    )
    from opencompass.configs.datasets.ruler.ruler_combined_gen import ruler_combined_datasets
    from opencompass.configs.summarizers.groups.ruler import ruler_summary_groups

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
models = internlm2_5_7b_chat_1m
work_dir = './outputs/ruler'


infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=2),
    runner=dict(
        type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLInferTask), retry=5
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=32, task=dict(type=OpenICLEvalTask)),
)

summarizer = dict(
    dataset_abbrs=['ruler_4k', 'ruler_8k', 'ruler_16k', 'ruler_32k'],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)
