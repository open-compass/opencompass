from mmengine.config import read_base

with read_base():
    from ...ceval.ceval_ppl_578f8d import ceval_datasets
    from ...agieval.agieval_mixed_713d14 import agieval_datasets
    from ...mmlu.mmlu_ppl_ac766d import mmlu_datasets
    from ...cmmlu.cmmlu_ppl_8b9c76 import cmmlu_datasets
    from ...GaokaoBench.GaokaoBench_mixed_9af5ee import GaokaoBench_datasets
    from ...ARC_c.ARC_c_gen_1e0de5 import ARC_c_datasets
    from ...ARC_e.ARC_e_gen_1e0de5 import ARC_e_datasets

    from ...SuperGLUE_WiC.SuperGLUE_WiC_ppl_312de9 import WiC_datasets
    from ...FewCLUE_chid.FewCLUE_chid_ppl_8f2872 import chid_datasets
    from ...CLUE_afqmc.CLUE_afqmc_ppl_6507d7 import afqmc_datasets
    from ...SuperGLUE_WSC.SuperGLUE_WSC_ppl_003529 import WSC_datasets
    from ...tydiqa.tydiqa_gen_978d2a import tydiqa_datasets
    from ...flores.flores_gen_806ede import flores_datasets

    from ...SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_314797 import BoolQ_datasets
    from ...commonsenseqa.commonsenseqa_ppl_5545e2 import commonsenseqa_datasets
    from ...triviaqa.triviaqa_gen_0356ec import triviaqa_datasets
    from ...nq.nq_gen_0356ec import nq_datasets

    from ...CLUE_C3.CLUE_C3_gen_8c358f import C3_datasets
    from ...race.race_ppl_5831a0 import race_datasets
    from ...obqa.obqa_gen_9069e4 import obqa_datasets
    from ...FewCLUE_csl.FewCLUE_csl_ppl_841b62 import csl_datasets
    from ...lcsts.lcsts_gen_8ee1fe import lcsts_datasets
    from ...Xsum.Xsum_gen_31397e import Xsum_datasets
    from ...FewCLUE_eprstmt.FewCLUE_eprstmt_gen_740ea0 import eprstmt_datasets
    from ...lambada.lambada_gen_217e11 import lambada_datasets

    from ...CLUE_cmnli.CLUE_cmnli_ppl_fdc6de import cmnli_datasets
    from ...CLUE_ocnli.CLUE_ocnli_gen_c4cb6c import ocnli_datasets
    from ...SuperGLUE_AX_b.SuperGLUE_AX_b_gen_4dfefa import AX_b_datasets
    from ...SuperGLUE_AX_g.SuperGLUE_AX_g_gen_68aac7 import AX_g_datasets
    from ...SuperGLUE_RTE.SuperGLUE_RTE_gen_68aac7 import RTE_datasets
    from ...SuperGLUE_COPA.SuperGLUE_COPA_gen_91ca53 import COPA_datasets
    from ...SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen_a69961 import ReCoRD_datasets
    from ...hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    from ...piqa.piqa_gen_1194eb import piqa_datasets
    from ...siqa.siqa_ppl_e8d8c5 import siqa_datasets
    from ...math.math_gen_265cce import math_datasets
    from ...gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from ...drop.deprecated_drop_gen_8a9ed9 import drop_datasets
    from ...humaneval.deprecated_humaneval_gen_a82cae import humaneval_datasets
    from ...mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets
    from ...bbh.bbh_gen_5bf00b import bbh_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
