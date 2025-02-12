from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.math.math_0shot_llm_judge_v2_gen_31d777 import \
        math_datasets
    # 选择一个感兴趣的模型
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_72b_instruct import \
        models as qwen2_5_72b_instruct_model

eval_model_name = 'eval_model_name'
postprocessor_model_name = 'postprocessor_model_name'
eval_model_urls = ['http://0.0.0.0:23333/v1']
postprocessor_model_urls = ['http://0.0.0.0:23333/v1']

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

for dataset in datasets:
    dataset['eval_cfg']['evaluator']['model_name'] = eval_model_name
    dataset['eval_cfg']['evaluator']['url'] = eval_model_urls
    dataset['eval_cfg']['evaluator']['post_url'] = postprocessor_model_urls
    dataset['eval_cfg']['evaluator'][
        'post_model_name'] = postprocessor_model_name

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
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner,
                max_num_workers=256,
                task=dict(type=OpenICLEvalTask)),
)
