from mmengine.config import read_base

with read_base():
    from .datasets.lveval.lveval import LVEval_datasets as datasets
    from .models.hf_llama.hf_llama2_7b_chat import models
    from .summarizers.lveval import summarizer

models[0][
    'path'
] = '/path/to/your/huggingface_models/Llama-2-7b-chat-hf'
models[0][
    'tokenizer_path'
] = '/path/to/your/huggingface_models/Llama-2-7b-chat-hf'
models[0]['max_seq_len'] = 4096
models[0]['generation_kwargs'] = dict(do_sample=False)
models[0]['mode'] = 'mid'  # truncate in the middle
