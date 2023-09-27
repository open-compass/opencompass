from opencompass.models.claude_api.claude_api import Claude
from opencompass.utils.text_postprocessors import last_option_postprocess, first_option_postprocess
from opencompass.models.claude_api.postprocessors import (yes_no_postprocess, humaneval_claude2_postprocess, record_postprocess,
                                                          gsm8k_postprocess, strategyqa_pred_postprocess, mbpp_postprocess,
                                                          lcsts_postprocess)


agieval_single_choice_sets = [
    'gaokao-chinese',
    'gaokao-english',
    'gaokao-geography',
    'gaokao-history',
    'gaokao-biology',
    'gaokao-chemistry',
    'gaokao-mathqa',
    'logiqa-zh',
    'lsat-ar',
    'lsat-lr',
    'lsat-rc',
    'logiqa-en',
    'sat-math',
    'sat-en',
    'sat-en-without-passage',
    'aqua-rat',
]
agieval_multiple_choices_sets = [
    'gaokao-physics',
    'jec-qa-kd',
    'jec-qa-ca',
]

claude_postprocessors = {
    'ceval-*': dict(type=last_option_postprocess, options='ABCD'),
    'bustm-*': dict(type=last_option_postprocess, options='AB'),
    'summedits': dict(type=last_option_postprocess, options='AB'),
    'WiC': dict(type=last_option_postprocess, options='AB'),
    'gsm8k': dict(type=gsm8k_postprocess),
    'openai_humaneval': dict(type=humaneval_claude2_postprocess),
    'lcsts': dict(type=lcsts_postprocess),
    'mbpp': dict(type=mbpp_postprocess),
    'strategyqa': dict(type=strategyqa_pred_postprocess),
    'WSC': dict(type=yes_no_postprocess),
    'BoolQ': dict(type=yes_no_postprocess),
    'cmnli': dict(type=first_option_postprocess, options='ABC'),
    'ocnli_fc-*': dict(type=first_option_postprocess, options='ABC'),
    'MultiRC': dict(type=yes_no_postprocess),
    'ReCoRD': dict(type=record_postprocess),
    'commonsense_qa': dict(type=last_option_postprocess, options='ABCDE'),
}

for _name in agieval_multiple_choices_sets + agieval_single_choice_sets:
    claude_postprocessors[f'agieval-{_name}'] = dict(type=last_option_postprocess, options='ABCDE')

models = [
    dict(abbr='Claude2',
        type=Claude,
        path='claude-2',
        key='YOUR_CLAUDE_KEY',
        query_per_second=1,
        max_out_len=2048, max_seq_len=2048, batch_size=2,
        pred_postprocessor=claude_postprocessors,
    ),
]
