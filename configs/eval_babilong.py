from opencompass.partitioners import (
    NaivePartitioner,
    NumWorkerPartitioner,
)
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # Models
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import (
        models as lmdeploy_internlm2_5_7b_chat_model,
    )
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import (
        models as lmdeploy_qwen2_5_7b_instruct_model,
    )
    from opencompass.configs.models.yi.lmdeploy_yi_1_5_9b_chat import (
        models as lmdeploy_yi_1_5_9b_chat_model,
    )
    from opencompass.configs.models.chatglm.lmdeploy_glm4_9b_chat import (
        models as lmdeploy_glm4_9b_chat_model,
    )
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_1_8b_instruct import (
        models as lmdeploy_llama3_1_8b_instruct_model,
    )
    from opencompass.configs.models.mistral.lmdeploy_ministral_8b_instruct_2410 import (
        models as lmdeploy_ministral_8b_instruct_2410_model,
    )

    # Datasets
    from opencompass.configs.datasets.babilong.babilong_0k_gen import (
        babiLong_0k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_4k_gen import (
        babiLong_4k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_16k_gen import (
        babiLong_16k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_32k_gen import (
        babiLong_32k_datasets,
    )

    from opencompass.configs.datasets.babilong.babilong_128k_gen import (
        babiLong_128k_datasets,
    )
    from opencompass.configs.datasets.babilong.babilong_256k_gen import (
        babiLong_256k_datasets,
    )
    from opencompass.configs.summarizers.groups.babilong import (
        babilong_summary_groups,
    )

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
for model in models:
    model['engine_config']['session_len'] = 1024 * 1024
    model['max_seq_len'] = 1024 * 1024
    model['engine_config']['tp'] = 4
    model['run_cfg']['num_gpus'] = 4


summarizer = dict(
    dataset_abbrs=[
        'babilong_0k',
        'babilong_2k',
        'babilong_4k',
        'babilong_16k',
        'babilong_32k',
        'babilong_128k',
        'babilong_256k',
        'babilong_1m',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)


# infer = dict(
#     partitioner=dict(type=NumWorkerPartitioner, num_worker=4),
#     runner=dict(
#         type=LocalRunner,
#         max_num_workers=16,
#         task=dict(type=OpenICLInferTask),
#         retry=5,
#     ),
# )

# eval = dict(
#     partitioner=dict(type=NaivePartitioner),
#     runner=dict(
#         type=LocalRunner, max_num_workers=32, task=dict(type=OpenICLEvalTask)
#     ),
# )

from opencompass.models.openai_api import OpenAI
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import DLCRunner, LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

# Modify the settings to your own settings!
python_env_path = '/cpfs01/user/xiaolinchen/miniconda3/envs/oc_lmdeploy'
# python_env_path = '/cpfs01/user/xiaolinchen/miniconda3/envs/oc_vllm'
dlc_config_path = '/cpfs01/user/xiaolinchen/.dlc/config'

############## A Cluster Settings ##############
data_sources = [
    'd-ixfoucd6qe8hbs7tis',  # public-cpfs
    'd-69a0iiiswaqy0iwlly',  # public-nas
    'd-7u8o8c7bznnaspwr7j',  # llmeval-cpfs
    'd-skzfy7p65wr7enm051',  # lleval-nas
    'd-f8ithls7kxoyip67oq',  # public-cpfs02 /cpfs02/puyu/shared/
    'd-mwdh56zjdzl3dpwf0c',  # public-cpfs02
    'd-xx4zxywl2xtv776n3d',  # puyu
    'd-bgq1qvvubru84hqy22',  # xiaolinchen-cpfs  CHANGE TO YOU OWN DATASET
]
worker_image = 'pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/zhangsongyang:compass-corebench1-6'
workspace_id = '86610' # llmeval_dlc
resource_id = 'quotaz819frn5q3u'
dlc_job_cmd = ''
############## A Cluster Settings End ##########


aliyun_cfg = dict(
    # users can also use own conda env
    python_env_path=python_env_path,
    dlc_config_path=dlc_config_path,
    workspace_id=workspace_id,
    worker_image=worker_image,
    resource_id=resource_id,
    data_sources=data_sources,
    hf_offline=True,
    # optional, suggest to set the http_proxy if `hf_offline` if False.
    # http_proxy="https://zhoufengzhe:gqjICYCei2y9OkqzfYmbhYB8UphnCA9clOKckTERO1fTBpp7GvYTukN2BEsC@aliyun-proxy.pjlab.org.cn:13128",
    # optional, using mirror to speed up the huggingface download
    hf_endpoint='https://hf-mirror.com',
    # optional, if not set, will use the default cache path
    huggingface_cache='/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub',
    torch_cache='/cpfs01/shared/public/public_hdd/llmeval/model_weights/torch',
    dlc_job_cmd=dlc_job_cmd,
    extra_envs=[
        # 'LD_LIBRARY_PATH=/cpfs01/shared/public/zhaoqian/cuda-compat-12-2:$LD_LIBRARY_PATH',
        'COMPASS_DATA_CACHE=/cpfs01/shared/public/llmeval/compass_data_cache',
        'TIKTOKEN_CACHE_DIR=/cpfs01/shared/public/public_hdd/llmeval/share_tiktoken',
    ],
)

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=DLCRunner,
        max_num_workers=64,
        retry=0,  # Modify if needed
        aliyun_cfg=aliyun_cfg,
        task=dict(type=OpenICLInferTask),
    ),
)


eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=DLCRunner,
        aliyun_cfg=aliyun_cfg,
        max_num_workers=128,
        retry=0,  # Modify if needed
        task=dict(type=OpenICLEvalTask),
    ),
)

work_dir = './outputs/babilong'
