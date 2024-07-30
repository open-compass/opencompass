
# export DATASET_SOURCE='ModelScope' # before run this script
from datasets import Dataset, DatasetDict
from mmengine.config import read_base
from tqdm import tqdm

with read_base():
    from .datasets.ceval.ceval_gen import ceval_datasets # ok
    from .datasets.ceval.ceval_clean_ppl import ceval_datasets as ceval_clean_datasets # ok

    from .datasets.mmlu.mmlu_gen import mmlu_datasets # ok
    from .datasets.mmlu.mmlu_clean_ppl import mmlu_datasets as mmlu_clean_datasets # ok
    from .datasets.cmmlu.cmmlu_gen import cmmlu_datasets # ok

    from .datasets.GaokaoBench.GaokaoBench_gen import GaokaoBench_datasets # ok
    from .datasets.GaokaoBench.GaokaoBench_mixed import GaokaoBench_datasets as GaokaoBench_mixed_datasets # ok
    from .datasets.GaokaoBench.GaokaoBench_no_subjective_gen_4c31db import GaokaoBench_datasets as GaokaoBench_no_subjective_datasets # ok

    from .datasets.humaneval.humaneval_gen import humaneval_datasets # ok
    from .datasets.humaneval.humaneval_repeat10_gen_8e312c import humaneval_datasets as humaneval_repeat10_datasets # ok


    from .datasets.commonsenseqa.commonsenseqa_gen import commonsenseqa_datasets # 额外处理gpt

    from .datasets.strategyqa.strategyqa_gen import strategyqa_datasets
    from .datasets.bbh.bbh_gen import bbh_datasets
    from .datasets.Xsum.Xsum_gen import Xsum_datasets
    from .datasets.winogrande.winogrande_gen import winogrande_datasets
    from .datasets.winogrande.winogrande_ll import winogrande_datasets as winogrande_ll_datasets  # ok
    from .datasets.winogrande.winogrande_5shot_ll_252f01 import winogrande_datasets as winogrande_5shot_ll_datasets # ok
    from .datasets.obqa.obqa_gen import obqa_datasets # ok
    from .datasets.obqa.obqa_ppl_6aac9e import obqa_datasets as obqa_ppl_datasets # ok
    from .datasets.agieval.agieval_gen import agieval_datasets as agieval_v2_datasets # ok
    from .datasets.agieval.agieval_gen_a0c741 import agieval_datasets as agieval_v1_datasets # ok

    from .datasets.siqa.siqa_gen import siqa_datasets as siqa_v2_datasets # ok
    from .datasets.siqa.siqa_gen_18632c import siqa_datasets as siqa_v3_datasets # ok
    from .datasets.siqa.siqa_ppl_42bc6e import siqa_datasets as siqa_ppl_datasets # ok
    from .datasets.storycloze.storycloze_gen import storycloze_datasets # ok
    from .datasets.storycloze.storycloze_ppl import storycloze_datasets as storycloze_ppl_datasets # ok
    from .datasets.summedits.summedits_gen import summedits_datasets as summedits_v2_datasets # ok

    from .datasets.hellaswag.hellaswag_gen import hellaswag_datasets as hellaswag_v2_datasets # ok
    from .datasets.hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets as hellaswag_ice_datasets # ok
    from .datasets.hellaswag.hellaswag_clean_ppl import hellaswag_datasets as hellaswag_clean_datasets # ok
    from .datasets.hellaswag.hellaswag_ppl_9dbb12 import hellaswag_datasets as hellaswag_v1_datasets # ok
    from .datasets.hellaswag.hellaswag_ppl_a6e128 import hellaswag_datasets as hellaswag_v3_datasets # ok

    from .datasets.mbpp.mbpp_gen import mbpp_datasets as mbpp_v1_datasets # ok
    from .datasets.mbpp.mbpp_passk_gen_830460 import mbpp_datasets as mbpp_v2_datasets # ok
    from .datasets.mbpp.sanitized_mbpp_gen_830460 import sanitized_mbpp_datasets # ok
    from .datasets.nq.nq_gen import nq_datasets # ok
    from .datasets.lcsts.lcsts_gen import lcsts_datasets # ok

    from .datasets.math.math_gen import math_datasets # ok
    from .datasets.piqa.piqa_gen import piqa_datasets as piqa_v2_datasets # ok
    from .datasets.piqa.piqa_ppl import piqa_datasets as piqa_v1_datasets # ok
    from .datasets.piqa.piqa_ppl_0cfff2 import piqa_datasets as piqa_v3_datasets # ok
    from .datasets.lambada.lambada_gen import lambada_datasets # ok
    from .datasets.tydiqa.tydiqa_gen import tydiqa_datasets # ok

    from .datasets.triviaqa.triviaqa_gen import triviaqa_datasets # ok
    from .datasets.triviaqa.triviaqa_wiki_1shot_gen_20a989 import triviaqa_datasets as triviaqa_wiki_1shot_datasets # ok

    from .datasets.CLUE_afqmc.CLUE_afqmc_gen import afqmc_datasets # ok
    from .datasets.CLUE_cmnli.CLUE_cmnli_gen import cmnli_datasets # ok
    from .datasets.CLUE_cmnli.CLUE_cmnli_ppl import cmnli_datasets as cmnli_ppl_datasets # ok
    from .datasets.CLUE_ocnli.CLUE_ocnli_gen import ocnli_datasets # ok

    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets # ok
    from .datasets.ARC_c.ARC_c_gen import ARC_c_datasets # ok
    from .datasets.ARC_c.ARC_c_clean_ppl import ARC_c_datasets as ARC_c_clean_datasets # ok
    from .datasets.ARC_e.ARC_e_gen import ARC_e_datasets # ok
    from .datasets.race.race_ppl import race_datasets # ok
    from .models.opt.hf_opt_125m import models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
for d in datasets:
    d['reader_cfg'].update({
        'train_range':'[0:5]',
        'test_range':'[0:5]'
    })
