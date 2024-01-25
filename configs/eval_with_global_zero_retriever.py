from mmengine.config import read_base
from opencompass.openicl.icl_retriever import ZeroRetriever

with read_base():
    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat_models
    from .datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets


datasets = [*hf_qwen_7b_chat_models]

models = [*mmlu_datasets]  # 5-shot


# retriever here will overwrite datasets' retriever and be applied to all tasks
# e.g. the retriever of mmlu_datasets above will be overwrite by ZeroRetriever
infer = dict(
    retriever=dict(type=ZeroRetriever),   
)
