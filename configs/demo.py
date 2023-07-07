from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

with read_base():
    # from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.winograd.winograd_ppl import winograd_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets

datasets = [*siqa_datasets]

# OPT-1.3b
opt1300m = dict(
       abbr='opt350m',          
       type=HuggingFaceCausalLM,
       path='facebook/opt-125m',
       tokenizer_path='facebook/opt-125m',
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True,
       ),
       max_out_len=100,
       max_seq_len=2048,
       batch_size=16,
       model_kwargs=dict(device_map='auto'),
       run_cfg=dict(num_gpus=1, num_procs=1),
    )

models = [opt1300m]
