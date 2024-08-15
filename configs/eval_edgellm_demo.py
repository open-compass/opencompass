from mmengine.config import read_base

with read_base():
    # datasets
    from .datasets.commonsenseqa.commonsenseqa_llama3_gen_734a22 import commonsenseqa_datasets
    from .datasets.longbench.longbench import longbench_datasets
    from .datasets.bbh.bbh_gen import bbh_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from .datasets.humaneval.humaneval_gen import humaneval_datasets
    from .datasets.FewCLUE_chid.FewCLUE_chid_gen import chid_datasets
    from .datasets.truthfulqa.truthfulqa_gen import truthfulqa_datasets
    # models
    from .models.hf_llama.hf_llama3_8b import models as hf_llama3_8b_model
    from .models.qwen.hf_qwen2_7b import models as hf_qwen2_7b_model
    
datasets = sum([v for k, v in locals().items() if k.endswith('_datasets') or k == 'datasets'], [])
work_dir = './outputs/edgellm/'
