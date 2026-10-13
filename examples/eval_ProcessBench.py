from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate

with read_base():
    from opencompass.configs.datasets.ProcessBench.processbench_gen import processbench_datasets as processbench_datasets

    from opencompass.configs.models.qwen2_5.vllm_qwen2_5_7b_instruct import models as vllm_qwen2_5_7b_instruct_model
    from opencompass.configs.models.qwen2_5.vllm_qwen2_5_14b_instruct import models as vllm_qwen2_5_14b_instruct_model
    from opencompass.configs.models.qwen2_5.vllm_qwen2_5_32b_instruct import models as vllm_qwen2_5_32b_instruct_model
    from opencompass.configs.models.qwen2_5.vllm_qwen2_5_72b_instruct import models as vllm_qwen2_5_72b_instruct_model
    # from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import models as lmdeploy_qwen2_5_7b_instruct_model
    # from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_72b_instruct import models as lmdeploy_qwen2_5_72b_instruct_model

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets') or k == 'datasets'], [])
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])


from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(type=OpenICLEvalTask)
    ),
)

work_dir = 'outputs/ProcessBench'
