from mmengine.config import read_base

with read_base():
    from .flores_gen_2697d7 import PMMEval_flores_datasets
    from .humanevalxl_gen_bdec92 import PMMEval_HumanEvalXL_datasets
    from .mgsm_gen_679720 import PMMEval_MGSM_datasets
    from .mhellaswag_gen_1a6b73 import PMMEval_MHellaswag_datasets
    from .mifeval_gen_79f8fb import PMMEval_MIFEval_datasets
    from .mlogiqa_gen_36c4f9 import PMMEval_MLogiQA_datasets
    from .mmmlu_gen_d5017d import PMMEval_MMMLU_datasets
    from .xnli_gen_973734 import PMMEval_XNLI_datasets


PMMEval_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
