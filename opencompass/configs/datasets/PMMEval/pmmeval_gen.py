from mmengine.config import read_base

with read_base():
    from .flores_gen_123abc import PMMEval_flores_datasets
    from .humanevalxl_gen_123abc import PMMEval_HumanEvalXL_datasets
    from .mgsm_gen_123abc import PMMEval_MGSM_datasets
    from .mhellaswag_gen_123abc import PMMEval_MHellaswag_datasets
    from .mifeval_gen_123abc import PMMEval_MIFEval_datasets
    from .mlogiqa_gen_123abc import PMMEval_MLogiQA_datasets
    from .mmmlu_gen_123abc import PMMEval_MMMLU_datasets
    from .xnli_gen_123abc import PMMEval_XNLI_datasets


PMMEval_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
