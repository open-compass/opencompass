from copy import deepcopy
from mmengine.config import read_base

with read_base():
    from .datasets.teval.teval_en_gen_1ac254 import teval_datasets as teval_en_datasets
    from .datasets.teval.teval_zh_gen_1ac254 import teval_datasets as teval_zh_datasets

    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat_model
    from .models.hf_internlm.hf_internlm2_chat_7b import models as hf_internlm2_chat_7b_model
    from .models.hf_llama.hf_llama2_7b_chat import models as hf_llama2_7b_chat_model

    from .summarizers.teval import summarizer

meta_template_system_patches = {
    'internlm2-chat-7b-hf': dict(role='SYSTEM', begin='<|im_start|>system\n', end='<|im_end|>\n'),
    'internlm2-chat-20b-hf': dict(role='SYSTEM', begin='<|im_start|>system\n', end='<|im_end|>\n'),
}

_origin_models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
models = []
for m in _origin_models:
    m = deepcopy(m)
    if 'meta_template' in m and 'round' in m['meta_template']:
        round = m['meta_template']['round']
        if all(r['role'].upper() != 'SYSTEM' for r in round):  # no system round
            if m['abbr'] in meta_template_system_patches:
                system_round = meta_template_system_patches[m['abbr']]
            else:
                system_round = [r for r in round if r['role'].upper() == 'HUMAN'][0]
                system_round = deepcopy(system_round)
                system_round['role'] = 'SYSTEM'
            m['meta_template']['round'].append(system_round)
    else:
        raise ValueError(f'no meta_template.round in {m.get("abbr", None)}')

    print(f'model {m["abbr"]} is using the following meta_template: {m["meta_template"]}')
    models.append(m)

datasets = teval_en_datasets + teval_zh_datasets
work_dir = './outputs/teval'


'''
dataset                                      version    metric          mode       qwen-7b-chat-hf    internlm2-chat-7b-hf    llama-2-7b-chat-hf
-------------------------------------------  ---------  --------------  -------  -----------------  ----------------------  --------------------
teval                                        -          naive_average   unknown              57.69                   78.18                 36.63
teval-instruct_v1                            10482d     string_metric   unknown              28.83                   98.08                 50.27
teval-instruct_v1                            10482d     json_metric     unknown              94.32                   97.08                  0.15
teval-plan_str_v1                            10482d     f1_score        unknown              66.24                   84.12                 45.72
teval-plan_json_v1                           10482d     f1_score        unknown              63.62                   77.71                 19.95
teval-reason_str_v1                          10482d     thought         unknown              54.14                   63.58                 44.92
teval-reason_retrieve_understand_json_v1     10482d     thought         unknown              33.77                   54.72                 21.49
teval-retrieve_str_v1                        10482d     name            unknown              73.89                   85.28                 60.6
teval-reason_retrieve_understand_json_v1     10482d     name            unknown              31.15                   68.97                 15.34
teval-understand_str_v1                      10482d     args            unknown              77.76                   93.03                 65.61
teval-reason_retrieve_understand_json_v1     10482d     args            unknown              44.16                   72.23                 26.84
teval-review_str_v1                          10482d     review_quality  unknown              62.22                   71.66                 44.35
teval_zh                                     -          naive_average   unknown              61.31                   75.01                 32.33
teval-instruct_v1_zh                         10482d     string_metric   unknown              88.69                   98.19                 23.64
teval-instruct_v1_zh                         10482d     json_metric     unknown              75.77                   96.62                  0.89
teval-plan_str_v1_zh                         10482d     f1_score        unknown              62.43                   70.69                 47.82
teval-plan_json_v1_zh                        10482d     f1_score        unknown              61.46                   68.95                 15.87
teval-reason_str_v1_zh                       10482d     thought         unknown              59.43                   68.14                 46.96
teval-reason_retrieve_understand_json_v1_zh  10482d     thought         unknown              39.19                   60.37                 23.91
teval-retrieve_str_v1_zh                     10482d     name            unknown              69.41                   84.22                 54.44
teval-reason_retrieve_understand_json_v1_zh  10482d     name            unknown              32.87                   70.46                 14.16
teval-understand_str_v1_zh                   10482d     args            unknown              84.39                   88.62                 77.29
teval-reason_retrieve_understand_json_v1_zh  10482d     args            unknown              48.71                   72.71                 28.83
teval-review_str_v1_zh                       10482d     review_quality  unknown              56.67                   60.57                 27.1
'''
