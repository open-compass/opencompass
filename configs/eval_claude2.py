from mmengine.config import read_base
from opencompass.models.claude_api.claude_api import Claude
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.utils.text_postprocessors import last_option_postprocess
from opencompass.models.claude_api.postprocessors import gsm8k_postprocess, humaneval_postprocess, lcsts_postprocess, mbpp_postprocess, strategyqa_pred_postprocess

with read_base():
    # choose a list of datasets
    from .datasets.collections.chat_medium import datasets
    # and output the results in a choosen format
    from .summarizers.medium import summarizer

models = [
    dict(abbr='Claude2',
        type=Claude,
        path='claude-2',
        key='YOUR_CLAUDE_KEY',
        query_per_second=1,
        max_out_len=2048, max_seq_len=2048, batch_size=2,
        pred_postprocessor={
            'agieval-*': dict(type=last_option_postprocess, options='ABCDE'),
            'ceval-*': dict(type=last_option_postprocess, options='ABCD'),
            'bustm-*': dict(type=last_option_postprocess, options='AB'),
            'hellaswag': dict(type=last_option_postprocess, options='ABCD'),
            'lukaemon_mmlu_*': dict(type=last_option_postprocess, options='ABCD'),
            'openbookqa': dict(type=last_option_postprocess, options='ABCD'),
            'piqa': dict(type=last_option_postprocess, options='AB'),
            'race-*': dict(type=last_option_postprocess, options='ABCD'),
            'summedits': dict(type=last_option_postprocess, options='AB'),
            'BoolQ': dict(type=last_option_postprocess, options='AB'),
            'CB': dict(type=last_option_postprocess, options='ABC'),
            'MultiRC': dict(type=last_option_postprocess, options='AB'),
            'RTE': dict(type=last_option_postprocess, options='AB'),
            'WiC': dict(type=last_option_postprocess, options='AB'),
            'WSC': dict(type=last_option_postprocess, options='AB'),
            'winogrande': dict(type=last_option_postprocess, options='AB'),
            'gsm8k': dict(type=gsm8k_postprocess),
            'openai_humaneval': dict(type=humaneval_postprocess),
            'lcsts': dict(type=lcsts_postprocess),
            'mbpp': dict(type=mbpp_postprocess),
            'strategyqa': dict(type=strategyqa_pred_postprocess),
        },
        ),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)),
)
