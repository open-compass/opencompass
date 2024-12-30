from mmengine.config import read_base
from opencompass.summarizers import DefaultSubjectiveSummarizer


with read_base():
    # 选择一个数据集列表
    from opencompass.configs.datasets.livestembench.livestembench_gen_3e3c50 import livestembench_datasets

    # 选择一个感兴趣的模型
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import models as qwen2_5_7b_instruct_lmdeploy_model
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_72b_instruct import models as qwen2_5_72b_instruct_lmdeploy_model

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])
models = [*qwen2_5_7b_instruct_lmdeploy_model, *qwen2_5_72b_instruct_lmdeploy_model]
summarizer = dict(type=DefaultSubjectiveSummarizer)

judge_models = qwen2_5_72b_instruct_lmdeploy_model


# -------------Inferen Stage ----------------------------------------

from opencompass.runners import LocalRunner
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask



infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)
    ),
)

sub_eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=[],
        judge_models=[],
    ),
    runner=dict(type=LocalRunner,
                max_num_workers=64,
                preemptible=True,
                retry=0,
                task=dict(type=SubjectiveEvalTask)),
)

eval = sub_eval
eval['partitioner']['models'] = models
eval['partitioner']['judge_models'] = judge_models

work_dir = './outputs/livestembench'
