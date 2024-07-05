from mmengine.config import read_base

with read_base():
    from .datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets
    from .datasets.agieval.agieval_mixed_713d14 import agieval_datasets
    from .datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .datasets.math.math_gen_265cce import math_datasets
    from .datasets.humaneval.deprecated_humaneval_gen_a82cae import humaneval_datasets
    from .datasets.mbpp.deprecated_sanitized_mbpp_gen_1e1056 import sanitized_mbpp_datasets

    from .models.hf_internlm.hf_internlm2_7b import models as hf_internlm2_7b_model
    from .models.hf_internlm.hf_internlm2_20b import models as hf_internlm2_20b_model

    from .summarizers.internlm2_keyset import summarizer

work_dir = './outputs/internlm2-keyset/'

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
