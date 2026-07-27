from copy import deepcopy

from mmengine.config import read_base

from opencompass.models import OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers import DefaultSummarizer
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.MMMU_Pro.MMMU_Pro_10c_vlmevalkit_gen import (  # noqa: E501
        mmmu_pro_datasets, )

vlmeval_eval_kwargs = dict(
    model='Intern-S2-Preview-FP8',
    api_base='https://token.pjlab.org.cn/v1/chat/completions',
    nproc=100,
    retry=3,
    timeout=600,
    temperature=0.0,
    max_tokens=100000,
)

datasets = deepcopy(mmmu_pro_datasets)
for dataset in datasets:
    dataset['eval_cfg']['evaluator']['eval_kwargs'] = vlmeval_eval_kwargs

models = [
    dict(type=OpenAISDK,
         abbr='Intern-S2-Preview-FP8-chat-completions',
         path='Intern-S2-Preview-FP8',
         key='ENV',
         openai_api_base='https://token.pjlab.org.cn/v1',
         tokenizer_path='gpt-4',
         image_format='JPEG',
         image_min_edge=100,
         include_reasoning_content=False,
         failure_message='Failed to obtain answer via API.',
         max_seq_len=128000,
         max_out_len=100000,
         batch_size=2,
         max_workers=100,
         query_per_second=20,
         temperature=0.0,
         retry=3,
         timeout=300)
]

infer = dict(partitioner=dict(
                type=NumWorkerPartitioner, 
                num_worker=30,
                num_split=30,
                min_task_size=1,
                force_rebuild=True,
            ),
             runner=dict(type=LocalRunner,
                         max_num_workers=30,
                         task=dict(type=OpenICLInferTask)))

eval = dict(partitioner=dict(type=NaivePartitioner, n=1),
            runner=dict(type=LocalRunner,
                        max_num_workers=30,
                        task=dict(type=OpenICLEvalTask)))

summarizer = dict(type=DefaultSummarizer)
work_dir = 'outputs/mmmu_pro_vlmevalkit'
