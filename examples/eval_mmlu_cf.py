from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu_cf.mmlu_cf_gen_040615 import \
        mmlu_cf_datasets
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_8b_instruct import \
        models as lmdeploy_llama3_8b_instruct_model
    from opencompass.configs.models.qwen2_5.hf_qwen2_5_7b_instruct import \
        models as hf_qwen2_5_7b_instruct_model
    from opencompass.configs.summarizers.mmlu_cf import summarizer

datasets = sum([
    v
    for k, v in locals().items() if k.endswith('_datasets') or k == 'datasets'
], [])
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner,
                max_num_workers=256,
                task=dict(type=OpenICLEvalTask)),
)

work_dir = 'outputs/debug/mmlu_cf'
