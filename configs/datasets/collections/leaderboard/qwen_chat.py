from mmengine.config import read_base

with read_base():
    from ...ceval.ceval_gen_5f30c7 import ceval_datasets
    from ...agieval.agieval_mixed_713d14 import agieval_datasets
    from ...mmlu.mmlu_gen_4d595a import mmlu_datasets
    from ...cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    from ...GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets
    from ...ARC_c.ARC_c_ppl_2ef631 import ARC_c_datasets
    from ...ARC_e.ARC_e_ppl_2ef631 import ARC_e_datasets

    from ...SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    from ...FewCLUE_chid.FewCLUE_chid_ppl_8f2872 import chid_datasets
    from ...CLUE_afqmc.CLUE_afqmc_gen_901306 import afqmc_datasets
    from ...SuperGLUE_WSC.SuperGLUE_WSC_ppl_003529 import WSC_datasets
    from ...tydiqa.tydiqa_gen_978d2a import tydiqa_datasets
    from ...flores.flores_gen_806ede import flores_datasets

    from ...SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_314797 import BoolQ_datasets
    from ...commonsenseqa.commonsenseqa_gen_c946f2 import commonsenseqa_datasets
    from ...triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from ...nq.nq_gen_c788f6 import nq_datasets

    from ...CLUE_C3.CLUE_C3_gen_8c358f import C3_datasets
    from ...race.race_gen_69ee4f import race_datasets
    from ...obqa.obqa_ppl_6aac9e import obqa_datasets
    from ...FewCLUE_csl.FewCLUE_csl_ppl_841b62 import csl_datasets
    from ...lcsts.lcsts_gen_8ee1fe import lcsts_datasets
    from ...Xsum.Xsum_gen_31397e import Xsum_datasets
    from ...FewCLUE_eprstmt.FewCLUE_eprstmt_ppl_f1e631 import eprstmt_datasets
    from ...lambada.lambada_gen_217e11 import lambada_datasets

    from ...CLUE_cmnli.CLUE_cmnli_ppl_fdc6de import cmnli_datasets
    from ...CLUE_ocnli.CLUE_ocnli_ppl_fdc6de import ocnli_datasets
    from ...SuperGLUE_AX_b.SuperGLUE_AX_b_ppl_6db806 import AX_b_datasets
    from ...SuperGLUE_AX_g.SuperGLUE_AX_g_ppl_66caf3 import AX_g_datasets
    from ...SuperGLUE_RTE.SuperGLUE_RTE_ppl_66caf3 import RTE_datasets
    from ...SuperGLUE_COPA.SuperGLUE_COPA_gen_91ca53 import COPA_datasets
    from ...SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen_30dea0 import ReCoRD_datasets
    from ...hellaswag.hellaswag_ppl_a6e128 import hellaswag_datasets
    from ...piqa.piqa_ppl_0cfff2 import piqa_datasets
    from ...siqa.siqa_ppl_e8d8c5 import siqa_datasets
    from ...math.math_gen_265cce import math_datasets
    from ...gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from ...drop.deprecated_drop_gen_8a9ed9 import drop_datasets
    from ...humaneval.deprecated_humaneval_gen_a82cae import humaneval_datasets
    from ...mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets
    from ...bbh.bbh_gen_5b92b0 import bbh_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
