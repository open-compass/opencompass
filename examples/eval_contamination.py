from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.ARC_c.ARC_c_clean_ppl import \
        ARC_c_datasets
    from opencompass.configs.datasets.ceval.ceval_clean_ppl import \
        ceval_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_clean_ppl import \
        hellaswag_datasets
    from opencompass.configs.datasets.mmlu.mmlu_clean_ppl import mmlu_datasets
    from opencompass.configs.models.hf_llama.hf_llama2_7b import \
        models as hf_llama2_7b_model
    from opencompass.configs.models.qwen.hf_qwen_7b import \
        models as hf_qwen_7b_model
    from opencompass.configs.models.yi.hf_yi_6b import models as hf_yi_6b_model
    from opencompass.configs.summarizers.contamination import summarizer

datasets = [
    *ceval_datasets, *mmlu_datasets, *hellaswag_datasets, *ARC_c_datasets
]
models = [*hf_yi_6b_model, *hf_qwen_7b_model, *hf_llama2_7b_model]
