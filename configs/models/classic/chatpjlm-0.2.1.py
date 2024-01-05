from opencompass.models import LLM
from mmengine.config import read_base

with read_base():
    from .._meta_templates import _template_0_1_0_ChatPJLM_v0_2_1rc1

models = [
    # https://aicarrier.feishu.cn/wiki/IlvrwXOCPiVCPAknYpJcnhjpn1f / SFT-v0.1.0
    dict(abbr='ChatPJLM-v0.2.1',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0426/sft_1006_mixture0425/2099/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc1,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8, num_procs=8)),
]
