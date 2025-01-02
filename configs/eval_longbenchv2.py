from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.runners import LocalRunner, VOLCRunner
from mmengine.config import read_base

with read_base():
    # from .volc import infer as volc_infer
    # from .volc import eval  as volc_eval
    # Models
    from opencompass.configs.models.chatglm.lmdeploy_glm4_9b_chat import (
        models as lmdeploy_glm4_9b_chat_model,
    )
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b_instruct import (
        models as lmdeploy_llama3_1_8b_instruct_model,
    )
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_70b_instruct import (
        models as lmdeploy_llama3_1_70b_instruct_model,
    )
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_3_70b_instruct import (
        models as lmdeploy_llama3_3_70b_instruct_model,
    )
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import (
        models as lmdeploy_qwen2_5_7b_instruct_model,
    )
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_72b_instruct import (
        models as lmdeploy_qwen2_5_72b_instruct_model,
    )
    from opencompass.configs.models.mistral.lmdeploy_mixtral_large_instruct_2407 import (
        models as lmdeploy_mixtral_large_instruct_2407_model,
    )
    from opencompass.configs.models.mistral.lmdeploy_mistral_large_instruct_2411 import (
        models as lmdeploy_mistral_large_instruct_2411_model,
    )
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat_1m import (
        models as lmdeploy_internlm2_5_7b_chat_1m_model,
    )

    # Datasets
    from opencompass.configs.datasets.longbench.longbenchv2.longbenchv2_gen import (
        LongBenchv2_datasets as LongBenchv2_datasets, 
    )

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

for model in models:
    model["max_seq_len"] = 2097152
    model["engine_config"]["session_len"] = 2097152
    # model["engine_config"]["rope_scaling_factor"]=2.5
    model["engine_config"]["tp"] = 2
    model["run_cfg"]["num_gpus"] = 2


# infer = volc_infer
# eval = volc_eval

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=8),
    runner=dict(
        type=LocalRunner, task=dict(type=OpenICLEvalTask), max_num_workers=32
    ),
)

work_dir = './outputs/longbenchv2'
