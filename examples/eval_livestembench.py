from mmengine.config import read_base

from opencompass.models import OpenAISDK

with read_base():
    # 选择一个数据集列表
    from opencompass.configs.datasets.livestembench.livestembench_gen_3e3c50 import \
        livestembench_datasets
    # 选择一个感兴趣的模型
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import \
        models as qwen2_5_7b_instruct_lmdeploy_model
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_72b_instruct import \
        models as qwen2_5_72b_instruct_lmdeploy_model

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])
models = [
    *qwen2_5_7b_instruct_lmdeploy_model, *qwen2_5_72b_instruct_lmdeploy_model
]

# Judge 模型配置
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

judge_cfg = dict(
    abbr='qwen2-5-72b-instruct',
    type=OpenAISDK,
    path='YOUR_SERVER_MODEL_NAME',  # 你的部署的模型名称
    key='None',
    openai_api_base=[
        'http://localhost:23333/v1',  # 你的模型部署的地址
    ],
    meta_template=api_meta_template,
    query_per_second=16,
    batch_size=16,
    temperature=0.001,
    max_completion_tokens=32768,
)

for dataset in datasets:
    dataset['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg

# -------------Inferen Stage ----------------------------------------

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
    partitioner=dict(type=NaivePartitioner, n=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(type=OpenICLEvalTask),
    ),
)

work_dir = './outputs/livestembench'
