from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.runners import LocalRunner, VOLCRunner
from mmengine.config import read_base

with read_base():

    # Models
    from opencompass.configs.models.chatglm.lmdeploy_glm4_9b_chat import (
        models as lmdeploy_glm4_9b_chat_model,
    )
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b_instruct import (
        models as lmdeploy_llama3_1_8b_instruct_model,
    )
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import (
        models as lmdeploy_qwen2_5_7b_instruct_model,
    )

    # Datasets
    from opencompass.configs.datasets.longbench.longbenchv2.longbenchv2_gen import (
        LongBenchv2_datasets as LongBenchv2_datasets,
    )

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

for model in models:
    model['max_seq_len'] = 1024 * 1024
    model['engine_config']['session_len'] = 1024 * 1024
    model['engine_config']['rope_scaling_factor'] = 2.5
    model['engine_config']['tp'] = 2
    model['run_cfg']['num_gpus'] = 2

work_dir = './outputs/longbenchv2'
