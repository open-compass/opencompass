from os import environ
from mmengine.config import read_base
from opencompass.models import ZhiPuV2AI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # from .datasets.collections.chat_medium import datasets
    from ..summarizers.medium import summarizer
    # from ..datasets.ceval.ceval_gen import ceval_datasets
    from ..datasets.commonsenseqa.commonsenseqa_gen import commonsenseqa_datasets
    # from ..datasets.GaokaoBench.GaokaoBench_gen import GaokaoBench_datasets
    # from ..datasets.strategyqa.strategyqa_gen import strategyqa_datasets
    # from ..datasets.bbh.bbh_gen import bbh_datasets
    # from ..datasets.Xsum.Xsum_gen import Xsum_datasets
    # from ..datasets.agieval.agieval_gen import agieval_datasets as agieval_v2_datasets
    # from ..datasets.gsm8k.gsm8k_gen import gsm8k_datasets as cur_datasets
    # from ..datasets.summedits.summedits_gen import summedits_datasets as cur_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

for d in datasets:
    d['reader_cfg'].update({
        'train_range':'[0:2]',
        'test_range':'[0:2]'
    })

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
        key=environ['ZHIPU_API_KEY'],
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
        max_num_workers=2,
        concurrent_users=2,
        task=dict(type=OpenICLInferTask),
        )
)

work_dir = 'outputs/api_zhipu_v2/'
