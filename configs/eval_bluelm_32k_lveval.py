from mmengine.config import read_base

with read_base():
    from .datasets.lveval.lveval import LVEval_datasets as datasets
    from .models.bluelm.hf_bluelm_7b_chat_32k import models
    from .summarizers.lveval import summarizer

models[0][
    'path'
] = '/path/to/your/huggingface_models/BlueLM-7B-Chat-32K'
models[0][
    'tokenizer_path'
] = '/path/to/your/huggingface_models/BlueLM-7B-Chat-32K'
models[0]['max_seq_len'] = 32768
models[0]['generation_kwargs'] = dict(do_sample=False)
models[0]['mode'] = 'mid'  # truncate in the middle
