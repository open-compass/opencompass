from mmengine.config import read_base

with read_base():
    from .models.qwen.hf_qwen_7b import models as hf_qwen_7b_base_models
    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat_models

    from .datasets.ceval.ceval_ppl_578f8d import ceval_datasets as base_ceval_datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets as chat_ceval_datasets

    from .internal.clusters.slurm import infer, eval
    # from .clusters.slurm import infer_split as infer, eval
    # from .clusters.slurm import infer_size as infer, eval
    # from .clusters.slurm import infer_size_split as infer, eval

base_ceval_datasets = base_ceval_datasets[:1]
chat_ceval_datasets = chat_ceval_datasets[-1:]

# If you do not want to run all the combinations of models and datasets, you
# can specify the combinations you want to run here. This is useful when you
# deleberately want to skip some subset of the combinations.
# Models and datasets in different combinations are recommended to be disjoint
# (different `abbr` in model & dataset configs), as we haven't tested this case
# throughly.
model_dataset_combinations = [
    dict(models=hf_qwen_7b_base_models, datasets=base_ceval_datasets),
    dict(models=hf_qwen_7b_chat_models, datasets=chat_ceval_datasets),
    # dict(models=[model_cfg1, ...], datasets=[dataset_cfg1, ...]),
]

# This union of models and datasets in model_dataset_combinations should be
# stored in the `models` and `datasets` variables below. Otherwise, modules
# like summarizer will miss out some information.
models = [*hf_qwen_7b_base_models, *hf_qwen_7b_chat_models]
datasets = [*base_ceval_datasets, *chat_ceval_datasets]

work_dir = './outputs/default/mdcomb/'

"""
dataset                 version    metric    mode    qwen-7b-hf    qwen-7b-chat-hf
----------------------  ---------  --------  ------  ------------  -----------------
ceval-computer_network  9b9417     accuracy  ppl     52.63         -
ceval-physician         6e277d     accuracy  gen     -             59.18
"""
