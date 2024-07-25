from mmengine.config import read_base
from opencompass.models import ZhiPuV2AI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # from .datasets.collections.chat_medium import datasets
    from ..summarizers.medium import summarizer
    from ..datasets.ceval.ceval_gen import ceval_datasets

datasets = [
    *ceval_datasets,
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
        abbr='glm4_notools',
        type=ZhiPuV2AI,
        path='glm-4',
        key='xxxxxx',
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
        batch_size=8)
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=2,
        concurrent_users=2,
        task=dict(type=OpenICLInferTask)),
)

work_dir = 'outputs/api_zhipu_v2/'
