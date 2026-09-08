from copy import deepcopy

from mmengine.config import read_base

from opencompass.models import OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers import DefaultSummarizer
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from opencompass.configs.datasets.MMMU_Pro.MMMU_Pro_10c_vlmevalkit_gen import (  # noqa: E501
        mmmu_pro_datasets, )

vlmeval_eval_kwargs = dict(
    model='qwen3-vl-8b-thinking',
    api_base=
    'http://example.com/v1/chat/completions',
    nproc=40,
    retry=10,
    timeout=3600,
    greedy=False,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
    presence_penalty=0.0,
    temperature=1.0,
    max_tokens=40960,
)

datasets = deepcopy(mmmu_pro_datasets)
for dataset in datasets:
    dataset['eval_cfg']['evaluator']['eval_kwargs'] = vlmeval_eval_kwargs

models = [
    dict(
        type=OpenAISDK,
        abbr='qwen3-vl-8b-thinking-chat-completions',
        path='qwen3-vl-8b-thinking',
        key='ENV',
        openai_api_base=
        'http://example.com/v1',
        tokenizer_path='gpt-4',
        image_format='JPEG',
        image_min_edge=100,
        max_seq_len=128000,
        max_out_len=40960,
        batch_size=40,
        max_workers=40,
        query_per_second=40,
        temperature=1.0,
        extra_body=dict(
            greedy=False,
            top_k=20,
            repetition_penalty=1.0,
        ),
        openai_extra_kwargs=dict(
            top_p=0.95,
            presence_penalty=0.0,
        ),
        retry=10,
        timeout=3600,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
        )
]

infer = dict(partitioner=dict(
    type=NumWorkerPartitioner,
    num_worker=4,
    num_split=4,
    min_task_size=1,
    force_rebuild=False,
),
             runner=dict(type=LocalRunner,
                         max_num_workers=4,
                         task=dict(type=OpenICLInferTask)))

eval = dict(partitioner=dict(type=NaivePartitioner, n=1),
            runner=dict(type=LocalRunner,
                        max_num_workers=4,
                        task=dict(type=OpenICLEvalTask)))

summarizer = dict(type=DefaultSummarizer)
work_dir = 'outputs/mmmu_pro_oc_qwen3-vl'
