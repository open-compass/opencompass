from copy import deepcopy

from mmengine.config import read_base

from opencompass.openicl.icl_retriever import ZeroRetriever

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import \
        mmlu_datasets  # this is a dataset evaluated with 5-shot
    from opencompass.configs.models.qwen.hf_qwen_7b_chat import models

datasets = []
for d in mmlu_datasets:
    d = deepcopy(d)
    d['infer_cfg']['retriever'] = dict(type=ZeroRetriever)
    datasets.append(d)
