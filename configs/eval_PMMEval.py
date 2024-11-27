from mmengine.config import read_base

from opencompass.models import HuggingFacewithChatTemplate


with read_base():
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import models

    # from opencompass.configs.datasets.PMMEval.flores_gen import PMMEval_flores_datasets
    # from opencompass.configs.datasets.PMMEval.humanevalxl_gen import PMMEval_HumanEvalXL_datasets
    # from opencompass.configs.datasets.PMMEval.mgsm_gen import PMMEval_MGSM_datasets
    # from opencompass.configs.datasets.PMMEval.mhellaswag_gen import PMMEval_MHellaswag_datasets
    # from opencompass.configs.datasets.PMMEval.mifeval_gen import PMMEval_MIFEval_datasets
    # from opencompass.configs.datasets.PMMEval.mlogiqa_gen import PMMEval_MLogiQA_datasets
    # from opencompass.configs.datasets.PMMEval.mmmlu_gen import PMMEval_MMMLU_datasets
    # from opencompass.configs.datasets.PMMEval.xnli import PMMEval_XNLI_datasets

    from opencompass.configs.datasets.PMMEval.pmmeval_gen import PMMEval_datasets

    from opencompass.configs.summarizers.PMMEval import summarizer


# datasets = PMMEval_flores_datasets
# datasets = PMMEval_HumanEvalXL_datasets
# datasets = PMMEval_MGSM_datasets
# datasets = PMMEval_MHellaswag_datasets
# datasets = PMMEval_MIFEval_datasets
# datasets = PMMEval_MLogiQA_datasets
# datasets = PMMEval_MMMLU_datasets
# datasets = PMMEval_XNLI_datasets

datasets = PMMEval_datasets
