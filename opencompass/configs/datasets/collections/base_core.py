from mmengine.config import read_base

with read_base():
    from ..mmlu.mmlu_ppl_ac766d import mmlu_datasets
    from ..cmmlu.cmmlu_ppl_041cbf import cmmlu_datasets
    from ..ceval.ceval_ppl_1cd8bf import ceval_datasets
    from ..GaokaoBench.GaokaoBench_no_subjective_gen_d21e37 import GaokaoBench_datasets
    from ..triviaqa.triviaqa_wiki_1shot_gen_20a989 import triviaqa_datasets
    from ..nq.nq_open_1shot_gen_20a989 import nq_datasets
    from ..race.race_ppl_abed12 import race_datasets
    from ..winogrande.winogrande_5shot_ll_252f01 import winogrande_datasets
    from ..hellaswag.hellaswag_10shot_ppl_59c85e import hellaswag_datasets
    from ..bbh.bbh_gen_98fba6 import bbh_datasets
    from ..gsm8k.gsm8k_gen_ee684f import gsm8k_datasets
    from ..math.math_evaluatorv2_gen_2f4a71 import math_datasets
    from ..TheoremQA.TheoremQA_post_v2_gen_2c2583 import TheoremQA_datasets
    from ..humaneval.deprecated_humaneval_gen_d2537e import humaneval_datasets
    from ..mbpp.deprecated_sanitized_mbpp_gen_cb43ef import sanitized_mbpp_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
