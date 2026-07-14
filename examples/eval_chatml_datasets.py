# flake8: noqa

from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():

    # Models (add your models here)
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
        models as hf_internlm2_5_7b_chat_model

    # Datasets
    from opencompass.configs.chatml_datasets.MaScQA.MaScQA_gen import datasets as MaScQA_chatml
    from opencompass.configs.chatml_datasets.CPsyExam.CPsyExam_gen import datasets as CPsyExam_chatml


models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

chatml_datasets = sum(
    (v for k, v in locals().items() if k.endswith('_chatml')),
    [],
)

# Your Judge Model Configs Here
judge_cfg = dict()

for dataset in chatml_datasets:
    if dataset['evaluator']['type'] == 'llm_evaluator':
        dataset['evaluator']['judge_cfg'] = judge_cfg
    if dataset['evaluator']['type'] == 'cascade_evaluator':
        dataset['evaluator']['llm_evaluator']['judge_cfg'] = judge_cfg

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

work_dir = 'outputs/ChatML_Datasets'