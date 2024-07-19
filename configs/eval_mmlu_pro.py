from mmengine.config import read_base

with read_base():
    from .datasets.mmlu_pro.mmlu_pro_gen_cdbebf import mmlu_pro_datasets

    from .models.qwen.lmdeploy_qwen2_7b_instruct import models as lmdeploy_qwen2_7b_instruct_model
    from .models.hf_llama.lmdeploy_llama3_8b_instruct import models as lmdeploy_llama3_8b_instruct_model

    from .summarizers.mmlu_pro import summarizer
    from .internal.clusters.local import infer_num_worker as infer
    from .internal.clusters.local import eval

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets') or k == 'datasets'], [])
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

work_dir = 'outputs/debug/mmlu_pro'

# dataset                    version    metric         mode      qwen2-7b-instruct-turbomind    llama-3-8b-instruct-turbomind
# -------------------------  ---------  -------------  ------  -----------------------------  -------------------------------
# mmlu_pro                   -          naive_average  gen                             46.18                            43.92
# mmlu_pro_biology           736233     accuracy       gen                             63.74                            64.02
# mmlu_pro_business          736233     accuracy       gen                             53.23                            46.01
# mmlu_pro_chemistry         736233     accuracy       gen                             35.25                            32.42
# mmlu_pro_computer_science  736233     accuracy       gen                             47.07                            44.88
# mmlu_pro_economics         736233     accuracy       gen                             59.00                            53.79
# mmlu_pro_engineering       736233     accuracy       gen                             26.73                            33.54
# mmlu_pro_health            736233     accuracy       gen                             47.31                            51.34
# mmlu_pro_history           736233     accuracy       gen                             42.78                            42.26
# mmlu_pro_law               736233     accuracy       gen                             28.07                            26.98
# mmlu_pro_math              736233     accuracy       gen                             53.59                            37.53
# mmlu_pro_philosophy        736233     accuracy       gen                             42.28                            42.48
# mmlu_pro_physics           736233     accuracy       gen                             39.11                            33.64
# mmlu_pro_psychology        736233     accuracy       gen                             60.90                            59.65
# mmlu_pro_other             736233     accuracy       gen                             47.40                            46.32
