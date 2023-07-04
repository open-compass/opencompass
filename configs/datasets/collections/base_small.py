from mmengine.config import read_base

with read_base():
    from ..ceval.ceval_ppl_275812 import ceval_datasets
    from ..bbh.bbh_gen_58abc3 import bbh_datasets
    from ..CLUE_CMRC.CLUE_CMRC_gen_72a8d5 import CMRC_datasets
    from ..CLUE_DRCD.CLUE_DRCD_gen_03b96b import DRCD_datasets
    from ..CLUE_afqmc.CLUE_afqmc_ppl_c83c36 import afqmc_datasets
    from ..FewCLUE_bustm.FewCLUE_bustm_ppl_47f2ab import bustm_datasets
    from ..FewCLUE_chid.FewCLUE_chid_ppl_b6cd88 import chid_datasets
    from ..FewCLUE_cluewsc.FewCLUE_cluewsc_ppl_2a9e61 import cluewsc_datasets
    from ..FewCLUE_eprstmt.FewCLUE_eprstmt_ppl_d3c387 import eprstmt_datasets
    from ..humaneval.humaneval_gen_d428f1 import humaneval_datasets
    from ..mbpp.mbpp_gen_4104e4 import mbpp_datasets
    from ..lambada.lambada_gen_7ffe3d import lambada_datasets
    from ..storycloze.storycloze_ppl_c1912d import storycloze_datasets
    from ..SuperGLUE_AX_b.SuperGLUE_AX_b_ppl_4bd960 import AX_b_datasets
    from ..SuperGLUE_AX_g.SuperGLUE_AX_g_ppl_8d9bf9 import AX_g_datasets
    from ..SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_f80fb0 import BoolQ_datasets
    from ..SuperGLUE_CB.SuperGLUE_CB_ppl_32adbb import CB_datasets
    from ..SuperGLUE_COPA.SuperGLUE_COPA_ppl_ddb78c import COPA_datasets
    from ..SuperGLUE_MultiRC.SuperGLUE_MultiRC_ppl_83a304 import MultiRC_datasets
    from ..SuperGLUE_RTE.SuperGLUE_RTE_ppl_29a22c import RTE_datasets
    from ..SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen_d8f19c import ReCoRD_datasets
    from ..SuperGLUE_WiC.SuperGLUE_WiC_ppl_4118db import WiC_datasets
    from ..SuperGLUE_WSC.SuperGLUE_WSC_ppl_85f45f import WSC_datasets
    from ..race.race_ppl_04e06a import race_datasets
    from ..math.math_gen_78bcba import math_datasets
    from ..gsm8k.gsm8k_gen_2dd372 import gsm8k_datasets
    from ..summedits.summedits_ppl_163352 import summedits_datasets
    from ..hellaswag.hellaswag_ppl_8e07d6 import hellaswag_datasets
    from ..piqa.piqa_ppl_788dbe import piqa_datasets
    from ..winogrande.winogrande_ppl_00f8ad import winogrande_datasets
    from ..obqa.obqa_ppl_2b5b12 import obqa_datasets
    from ..nq.nq_gen_c00b89 import nq_datasets
    from ..triviaqa.triviaqa_gen_cc3cbf import triviaqa_datasets
    from ..crowspairs.crowspairs_ppl_f60797 import crowspairs_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
