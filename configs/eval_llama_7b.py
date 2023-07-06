from mmengine.config import read_base

with read_base():
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets


datasets = [*piqa_datasets, *siqa_datasets]

from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        path='huggyllama/llama-7b',
        tokenizer_path='huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        abbr='llama-7b',            
        max_out_len=100,            
        batch_size=16,
        run_cfg=dict(num_gpus=1),   
    )
]