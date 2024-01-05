from opencompass.models import LLM
from mmengine.config import read_base

with read_base():
    from .._meta_templates import _template_0_1_0_ChatPJLM_v0_2_2rc5
    from .report_1 import models

models = models + [
    # https://aicarrier.feishu.cn/wiki/YutBwecBEi1D1ckYAgmcr7dTnth
    dict(abbr='ChatPJLM-v0.2.2rc5',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0511/sft_1006_v022rc5/1599/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_2rc5,
        max_out_len=100, max_seq_len=2048, batch_size=4, run_cfg=dict(num_gpus=8, num_procs=8)),
]
