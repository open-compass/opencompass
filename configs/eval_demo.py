from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

from mmengine.config import read_base

with read_base():
    from .datasets.piqa.piqa_ppl import piqa_datasets

datasets = piqa_datasets

models = [
    # OPT-1.3b
    dict(
       type=HuggingFaceCausalLM,
       path='facebook/opt-1.3b',
       tokenizer_path='facebook/opt-1.3b',
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
]
