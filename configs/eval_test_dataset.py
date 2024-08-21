from mmengine.config import read_base
from opencompass.models import LmdeployPytorchModel, TurboMindModel

with read_base():
    # choose a list of datasets
    from .datasets.ceval.ceval_gen_5f30c7 import \
        ceval_datasets  # noqa: F401, E501
    from .datasets.crowspairs.crowspairs_gen_381af0 import \
        crowspairs_datasets  # noqa: F401, E501
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import \
        gsm8k_datasets  # noqa: F401, E501
    from .datasets.mmlu.mmlu_gen_a484b3 import \
        mmlu_datasets  # noqa: F401, E501
    from .datasets.race.race_gen_69ee4f import \
        race_datasets  # noqa: F401, E501
    from .datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import \
        WiC_datasets  # noqa: F401, E501
    from .datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_7902a7 import \
        WSC_datasets  # noqa: F401, E501
    from .datasets.triviaqa.triviaqa_gen_2121ce import \
        triviaqa_datasets  # noqa: F401, E501
    


    from .datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from .datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    from .datasets.ceval.ceval_gen_2daf24 import ceval_datasets
    from .datasets.GaokaoBench.GaokaoBench_no_subjective_gen_4c31db import GaokaoBench_datasets
    from .datasets.triviaqa.triviaqa_wiki_1shot_gen_eaf81e import triviaqa_datasets
    from .datasets.nq.nq_open_1shot_gen_01cf41 import nq_datasets
    from .datasets.race.race_gen_69ee4f import race_datasets
    from .datasets.winogrande.winogrande_5shot_gen_b36770 import winogrande_datasets
    from .datasets.hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets
    from .datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .datasets.math.math_0shot_gen_393424 import math_datasets
    from .datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import TheoremQA_datasets
    from .datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.sanitized_mbpp_gen_a0fc46 import sanitized_mbpp_datasets
    from .datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets
    from .datasets.crowspairs.crowspairs_gen_381af0 import crowspairs_datasets


    
