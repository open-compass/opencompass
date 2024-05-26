from copy import deepcopy
from mmengine.config import read_base

with read_base():
    from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from .datasets.agieval.agieval_gen_64afd3 import agieval_datasets
    from .datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .datasets.math.math_evaluatorv2_gen_cecb31 import math_datasets
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.deprecated_sanitized_mbpp_gen_1e1056 import sanitized_mbpp_datasets

    from .models.hf_internlm.hf_internlm2_chat_7b import models as hf_internlm2_chat_7b_model
    from .models.hf_internlm.hf_internlm2_chat_20b import models as hf_internlm2_chat_20b_model

    from .summarizers.internlm2_keyset import summarizer

work_dir = './outputs/internlm2-chat-keyset/'

_origin_datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])
_origin_models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

_vanilla_datasets = [deepcopy(d) for d in _origin_datasets]
_vanilla_models = []
for m in _origin_models:
    m = deepcopy(m)
    if 'meta_template' in m and 'round' in m['meta_template']:
        round = m['meta_template']['round']
        if any(r['role'] == 'SYSTEM' for r in round):
            new_round = [r for r in round if r['role'] != 'SYSTEM']
            print(f'WARNING: remove SYSTEM round in meta_template for {m.get("abbr", None)}')
            m['meta_template']['round'] = new_round
    _vanilla_models.append(m)


datasets = _vanilla_datasets
models = _vanilla_models
