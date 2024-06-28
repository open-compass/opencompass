from mmengine.config import read_base

with read_base():
    from ..datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from ..datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    from ..datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from ..datasets.GaokaoBench.GaokaoBench_no_subjective_gen_4c31db import GaokaoBench_datasets
    from ..datasets.triviaqa.triviaqa_wiki_1shot_gen_bc5f21 import triviaqa_datasets
    from ..datasets.nq.nq_open_1shot_gen_2e45e5 import nq_datasets
    from ..datasets.race.race_gen_69ee4f import race_datasets
    from ..datasets.winogrande.winogrande_5shot_gen_b36770 import winogrande_datasets
    from ..datasets.hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets
    from ..datasets.bbh.bbh_gen_2879b0 import bbh_datasets
    from ..datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from ..datasets.math.math_0shot_gen_393424 import math_datasets
    from ..datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import TheoremQA_datasets
    from ..datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from ..datasets.mbpp.sanitized_mbpp_gen_830460 import sanitized_mbpp_datasets
    from ..datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets
    from ..datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
