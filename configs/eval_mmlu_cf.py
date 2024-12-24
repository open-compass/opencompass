from mmengine.config import read_base

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

# dataset                    version    metric         mode      qwen2-7b-instruct-turbomind    llama-3-8b-instruct-turbomind
# -------------------------  ---------  -------------  ------  -----------------------------  -------------------------------
# mmlu_cf                   -          naive_average  gen                             46.18                            43.92
# mmlu_cf_biology           736233     accuracy       gen                             63.74                            64.02
# mmlu_cf_business          736233     accuracy       gen                             53.23                            46.01
# mmlu_cf_chemistry         736233     accuracy       gen                             35.25                            32.42
# mmlu_cf_computer_science  736233     accuracy       gen                             47.07                            44.88
# mmlu_cf_economics         736233     accuracy       gen                             59.00                            53.79
# mmlu_cf_engineering       736233     accuracy       gen                             26.73                            33.54
# mmlu_cf_health            736233     accuracy       gen                             47.31                            51.34
# mmlu_cf_history           736233     accuracy       gen                             42.78                            42.26
# mmlu_cf_law               736233     accuracy       gen                             28.07                            26.98
# mmlu_cf_math              736233     accuracy       gen                             53.59                            37.53
# mmlu_cf_philosophy        736233     accuracy       gen                             42.28                            42.48
# mmlu_cf_physics           736233     accuracy       gen                             39.11                            33.64
# mmlu_cf_psychology        736233     accuracy       gen                             60.90                            59.65
# mmlu_cf_other             736233     accuracy       gen                             47.40                            46.32
