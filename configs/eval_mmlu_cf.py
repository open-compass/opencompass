from mmengine.config import read_base
import pdb
with read_base():
    from opencompass.configs.datasets.mmlu_cf.mmlu_cf_gen import mmlu_cf_datasets

    from opencompass.configs.models.qwen.lmdeploy_qwen2_7b_instruct import models as lmdeploy_qwen2_7b_instruct_model
    from opencompass.configs.models.hf_llama.lmdeploy_llama3_8b_instruct import models as lmdeploy_llama3_8b_instruct_model

    from opencompass.configs.summarizers.mmlu_cf import summarizer
    from opencompass.configs.internal.clusters.local import infer_num_worker as infer
    from opencompass.configs.internal.clusters.local import eval

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets') or k == 'datasets'], [])
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

work_dir = 'outputs/debug/mmlu_cf'
