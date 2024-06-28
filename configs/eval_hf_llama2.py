from mmengine.config import read_base

with read_base():
    from .datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets
    from .datasets.triviaqa.triviaqa_wiki_gen_d18bf4 import triviaqa_datasets
    from .datasets.nq.nq_open_gen_e93f8a import nq_datasets
    from .datasets.gsm8k.gsm8k_gen_3309bd import gsm8k_datasets
    from .datasets.humaneval.deprecated_humaneval_gen_a82cae import humaneval_datasets
    from .datasets.agieval.agieval_mixed_713d14 import agieval_datasets
    from .datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_314797 import BoolQ_datasets
    from .datasets.hellaswag.hellaswag_ppl_a6e128 import hellaswag_datasets
    from .datasets.obqa.obqa_ppl_6aac9e import obqa_datasets
    from .datasets.winogrande.winogrande_ll_c5cf57 import winogrande_datasets
    from .models.hf_llama.hf_llama2_7b import models
    from .summarizers.example import summarizer

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets') or k == 'datasets'], [])
work_dir = './outputs/llama2/'
