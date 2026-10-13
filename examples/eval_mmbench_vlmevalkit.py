from copy import deepcopy

from mmengine.config import read_base

from opencompass.models import OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers import DefaultSummarizer
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from opencompass.configs.datasets.MMBench.MMBench_DEV_EN_vlmevalkit_gen import (  # noqa: E501
        mmbench_datasets, )

vlmeval_eval_kwargs = dict(
    model='kimi-k2.6',
    api_base='https://example.com/v1/chat/completions',
    nproc=4,
    retry=3,
    timeout=600,
    temperature=0.0,
    max_tokens=32768,
)

datasets = deepcopy(mmbench_datasets)
for dataset in datasets:
    dataset['eval_cfg']['evaluator']['eval_kwargs'] = vlmeval_eval_kwargs

models = [
    dict(type=OpenAISDK,
         abbr='kimi-k2.6-chat-completions',
         path='kimi-k2.6',
         key='ENV',
         openai_api_base='https://example.com/v1',
         tokenizer_path='gpt-4',
         image_format='JPEG',
         image_min_edge=100,
         max_seq_len=128000,
         max_out_len=32768,
         batch_size=64,
         max_workers=4,
         query_per_second=3,
         temperature=0.0,
         retry=3,
         timeout=3600,
         pred_postprocessor=dict(type=extract_non_reasoning_content),
         )
]

infer = dict(partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
             runner=dict(type=LocalRunner,
                         max_num_workers=1,
                         task=dict(type=OpenICLInferTask)))

eval = dict(partitioner=dict(type=NaivePartitioner, n=1),
            runner=dict(type=LocalRunner,
                        max_num_workers=1,
                        task=dict(type=OpenICLEvalTask)))

summarizer = dict(type=DefaultSummarizer)
work_dir = 'outputs/mmbench_vlmevalkit'
