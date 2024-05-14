from mmengine.config import read_base

with read_base():
    from .dataset_collections.chat_OC15 import datasets

    from .models.hf_llama.hf_llama3_8b_instruct import models as hf_llama3_8b_instruct_model

    from .summarizers.chat_OC15 import summarizer


work_dir = 'outputs/debug/llama3-instruct'

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

# dataset               version    metric                        mode    llama-3-8b-instruct-hf
# --------------------  ---------  ----------------------------  ------  ------------------------
# average               -          naive_average                 gen     55.64
# mmlu                  -          naive_average                 gen     68.30
# cmmlu                 -          naive_average                 gen     53.29
# ceval                 -          naive_average                 gen     52.32
# GaokaoBench           -          weighted_average              gen     45.91
# triviaqa_wiki_1shot   eaf81e     score                         gen     79.01
# nq_open_1shot         01cf41     score                         gen     30.25
# race-high             9a54b6     accuracy                      gen     81.22
# winogrande            b36770     accuracy                      gen     66.46
# hellaswag             e42710     accuracy                      gen     74.33
# bbh                   -          naive_average                 gen     67.25
# gsm8k                 1d7fe4     accuracy                      gen     79.08
# math                  393424     accuracy                      gen     27.78
# TheoremQA             6f0af8     score                         gen     19.50
# openai_humaneval      8e312c     humaneval_pass@1              gen     55.49
# sanitized_mbpp        830460     score                         gen     66.54
# GPQA_diamond          4baadb     accuracy                      gen     25.76
# IFEval                3321a3     Prompt-level-strict-accuracy  gen     67.84
#                       -          -                             -       -
# mmlu                  -          naive_average                 gen     68.30
# mmlu-stem             -          naive_average                 gen     57.92
# mmlu-social-science   -          naive_average                 gen     77.83
# mmlu-humanities       -          naive_average                 gen     71.20
# mmlu-other            -          naive_average                 gen     71.79
# cmmlu                 -          naive_average                 gen     53.29
# cmmlu-stem            -          naive_average                 gen     45.40
# cmmlu-social-science  -          naive_average                 gen     54.63
# cmmlu-humanities      -          naive_average                 gen     54.14
# cmmlu-other           -          naive_average                 gen     59.52
# cmmlu-china-specific  -          naive_average                 gen     49.33
# ceval                 -          naive_average                 gen     52.32
# ceval-stem            -          naive_average                 gen     48.16
# ceval-social-science  -          naive_average                 gen     57.50
# ceval-humanities      -          naive_average                 gen     53.26
# ceval-other           -          naive_average                 gen     54.26
# ceval-hard            -          naive_average                 gen     35.59
