# Copyright (c) Alibaba, Inc. and its affiliates.
from mmengine.config import read_base
from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers import DefaultSummarizer
from opencompass.tasks import OpenICLInferTask


SWIFT_DEPLOY_API = 'http://127.0.0.1:8000/v1/chat/completions'

# TODO: BY JASON ONLY FOR TEST!
with read_base():
    from .datasets.ceval.ceval_gen import ceval_datasets


datasets = [*ceval_datasets]


api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)


models = [
    dict(abbr='LlaMA-3-8B-INSTRUCT',
         type=OpenAI,
         path='llama3-8b-instruct',
         # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
         key='your_openai_api_key',     # No need for swift deployment API
         meta_template=api_meta_template,
         query_per_second=1,
         max_out_len=2048,
         max_seq_len=2048,
         batch_size=1,
         run_cfg=dict(num_gpus=0),
         openai_api_base=SWIFT_DEPLOY_API,
         ),
]


infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=4,
        task=dict(type=OpenICLInferTask)),
)


summarizer = dict(
    dataset_abbrs=[
        ['ceval-test', 'naive_average'],

        # ['mmlu', 'naive_average'],
        # ['cmmlu', 'naive_average'],
        # ['GaokaoBench', 'weighted_average'],
        # ['triviaqa_wiki_1shot', 'score'],
        # ['nq_open_1shot', 'score'],
        # ['race-high', 'accuracy'],
        # ['winogrande', 'accuracy'],
        # ['hellaswag', 'accuracy'],
        # ['bbh', 'naive_average'],
        # ['gsm8k', 'accuracy'],
        # ['math', 'accuracy'],
        # ['TheoremQA', 'accuracy'],
        # ['openai_humaneval', 'humaneval_pass@1'],
        # ['sanitized_mbpp', 'score'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
