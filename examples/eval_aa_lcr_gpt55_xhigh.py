from copy import deepcopy

from mmengine.config import read_base

from opencompass.models import OpenAISDK, OpenAISDKResponse
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from opencompass.configs.datasets.aa_lcr.aa_lcr_gen import \
        aa_lcr_datasets


AA_LCR_NUM_RUNS = 3
AA_LCR_MAX_OUT_LEN = 128000
AA_LCR_MAX_SEQ_LEN = 1050000
AA_LCR_TEST_RANGE = None

work_dir = './output/aa_lcr_gpt55_xhigh'

datasets = deepcopy(aa_lcr_datasets)
for dataset in datasets:
    dataset['n'] = AA_LCR_NUM_RUNS
    dataset['eval_cfg']['evaluator']['dataset_cfg']['n'] = 1

    if AA_LCR_TEST_RANGE:
        dataset['reader_cfg']['test_range'] = AA_LCR_TEST_RANGE
        dataset['eval_cfg']['evaluator']['dataset_cfg']['reader_cfg'][
            'test_range'] = AA_LCR_TEST_RANGE

    judge_cfg = dict(
        abbr='qwen3-235b-a22b-instruct-2507',
        type=OpenAISDK,
        path='qwen3-235b-a22b-instruct-2507',
        key='ENV',
        query_per_second=1,
        batch_size=10,
        temperature=0,
        tokenizer_path='gpt-4o-2024-05-13',
        max_out_len=1024,
        max_seq_len=131072,
        retry=5,
        timeout=3600,
    )
    dataset['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg

models = [
    dict(
        abbr='gpt-5.5-xhigh',
        type=OpenAISDKResponse,
        path='gpt-5.5',
        key='ENV',
        query_per_second=1,
        batch_size=100,
        tokenizer_path='gpt-4o-2024-05-13',
        max_out_len=AA_LCR_MAX_OUT_LEN,
        max_seq_len=AA_LCR_MAX_SEQ_LEN,
        max_workers=20,
        retry=5,
        timeout=3600,
        openai_extra_kwargs=dict(reasoning=dict(effort='xhigh')),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=10,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=10,
        task=dict(type=OpenICLEvalTask),
    ),
)
