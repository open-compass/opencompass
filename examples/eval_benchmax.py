from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.xlivecodebench.xlivecodebench_gen import \
        LCB_datasets
    from opencompass.configs.datasets.xgpqa.xgpqa_gen import \
        gpqa_datasets
    from opencompass.configs.datasets.xIFEval.xIFeval_gen import \
        xifeval_datasets
    from opencompass.configs.models.hf_llama.hf_llama3_8b_instruct import \
        models as hf_llama3_8b_instruct_models
    
datasets = [
    *LCB_datasets,
    *gpqa_datasets,
    *xifeval_datasets
]
models = [
    *hf_llama3_8b_instruct_models
]
