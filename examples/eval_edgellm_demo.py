from mmengine.config import read_base

with read_base():
    # datasets
    from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets
    from opencompass.configs.datasets.commonsenseqa.commonsenseqa_7shot_cot_gen_734a22 import \
        commonsenseqa_datasets
    from opencompass.configs.datasets.FewCLUE_chid.FewCLUE_chid_gen import \
        chid_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import \
        humaneval_datasets
    from opencompass.configs.datasets.longbench.longbench import \
        longbench_datasets
    from opencompass.configs.datasets.truthfulqa.truthfulqa_gen import \
        truthfulqa_datasets
    # models
    from opencompass.configs.models.hf_llama.hf_llama3_8b import \
        models as hf_llama3_8b_model
    from opencompass.configs.models.others.hf_phi_2 import \
        models as hf_phi_2_model
    from opencompass.configs.models.qwen.hf_qwen2_7b import \
        models as hf_qwen2_7b_model

datasets = sum([
    v
    for k, v in locals().items() if k.endswith('_datasets') or k == 'datasets'
], [])
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
work_dir = './outputs/edgellm/'

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# dataset                                      version    metric            mode      phi-2_hf
# -------------------------------------------  ---------  ----------------  ------  ----------
# commonsense_qa                               c946f2     accuracy          gen          65.19
# openai_humaneval                             8e312c     humaneval_pass@1  gen          30.49
# truthful_qa                                  5ddc62     rouge_max         gen           0.08
# truthful_qa                                  5ddc62     rouge_diff        gen          -0.00
# truthful_qa                                  5ddc62     rouge_acc         gen           0.41
# gsm8k                                        1d7fe4     accuracy          gen          62.40
# chid-dev                                     211ee7     accuracy          gen          12.87
# chid-test                                    211ee7     accuracy          gen          14.34
# bbh                                          -          naive_average     gen          59.50

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# dataset                                      version    metric            mode      Meta-Llama-3-8B_hf
# -------------------------------------------  ---------  ----------------  ------  --------------------
# commonsense_qa                               c946f2     accuracy          gen                     70.11
# openai_humaneval                             8e312c     humaneval_pass@1  gen                    26.22
# truthful_qa                                  5ddc62     rouge_max         gen                     0.07
# truthful_qa                                  5ddc62     rouge_diff        gen                    -0.01
# truthful_qa                                  5ddc62     rouge_acc         gen                     0.41
# gsm8k                                        1d7fe4     accuracy          gen                    55.80
# chid-dev                                     211ee7     accuracy          gen                    40.59
# chid-test                                    211ee7     accuracy          gen                    36.66
# bbh                                          -          naive_average     gen                    61.62
# 20240816_060452
# tabulate format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# dataset         version    metric      mode      qwen2-7b-hf
# --------------  ---------  ----------  ------  -------------
# commonsense_qa  734a22     accuracy    gen             65.19
# truthful_qa     5ddc62     rouge_max   gen              0.08
# truthful_qa     5ddc62     rouge_diff  gen             -0.02
# truthful_qa     5ddc62     rouge_acc   gen              0.44
