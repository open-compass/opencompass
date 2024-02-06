from mmengine.config import read_base
from opencompass.openicl.icl_retriever import ZeroRetriever
from copy import deepcopy

with read_base():
    from .models.qwen.hf_qwen_7b_chat import models
    from .datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets # this is a dataset evaluated with 5-shot


datasets = []
for d in mmlu_datasets:
    d = deepcopy(d)
    d['infer_cfg']['retriever'] = dict(type=ZeroRetriever)
    datasets.append(d)
