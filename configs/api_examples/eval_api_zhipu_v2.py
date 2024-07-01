from os import environ
from mmengine.config import read_base
from opencompass.models import ZhiPuV2AI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # from .datasets.collections.chat_medium import datasets
    from ..summarizers.medium import summarizer
    # from ..datasets.ceval.ceval_gen import ceval_datasets as cur_datasets
    # from ..datasets.ceval.ceval_clean_ppl import ceval_datasets as cur_datasets
    # from ..datasets.GaokaoBench.GaokaoBench_gen import GaokaoBench_datasets as cur_datasets
    # from ..datasets.gsm8k.gsm8k_gen import gsm8k_datasets as cur_datasets
    from ..datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_gen import AX_b_datasets as cur_datasets

datasets = [
    *cur_datasets,
]

# needs a special postprocessor for all
# except 'gsm8k' and 'strategyqa'
from opencompass.utils import general_eval_wrapper_postprocess
for _dataset in datasets:
    if _dataset['abbr'] not in ['gsm8k', 'strategyqa']:
        if hasattr(_dataset['eval_cfg'], 'pred_postprocessor'):
            _dataset['eval_cfg']['pred_postprocessor']['postprocess'] = _dataset['eval_cfg']['pred_postprocessor']['type']
            _dataset['eval_cfg']['pred_postprocessor']['type'] = general_eval_wrapper_postprocess
        else:
            _dataset['eval_cfg']['pred_postprocessor'] = {'type': general_eval_wrapper_postprocess}


api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
     dict(
        abbr='glm-4-flash',
        type=ZhiPuV2AI,
        path='glm-4-flash',
        key=environ["ZHIPU_API_KEY"],
        generation_kwargs={
            'tools': [
                {
                    'type': 'web_search',
                    'web_search': {
                        'enable': False # turn off the search
                    }
                }
            ]
        },
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=1)
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=1,
        concurrent_users=1,
        task=dict(type=OpenICLInferTask),
        )
)

work_dir = 'outputs/api_zhipu_v2/'
