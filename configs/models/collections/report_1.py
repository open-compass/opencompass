from opencompass.models import LLM
from mmengine.config import read_base

with read_base():
    from .._meta_templates import (_template_0_1_0_ChatPJLM_v0_2_1rc2, 
                                   _template_0_1_0_ChatPJLM_v0_2_1rc1,
                                    _template_0_1_0_ChatPJLM_v0_0_1)

models = [
    # PJLM-0.2 / 1006-15760
    dict(abbr="PJLM-0.2",
        type=LLM, path='model_weights:s3://model_weights/0331/1006/15760',
        tokenizer_path='/mnt/petrelfs/share_data/zhengmiao/llamav4.model', tokenizer_type='v4',
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8, num_procs=8)),
    # https://aicarrier.feishu.cn/wiki/PszSwVtt0i2VJ5kdDl4cIDTunDf / SFT-v0.0.1
    dict(abbr='ChatPJLM-v0.0.1',
        type=LLM, path='anonymous_ssd:s3://model_weights/0331/sft_1006_v3/4999',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_0_1,
        max_out_len=100, max_seq_len=4096, batch_size=8, run_cfg=dict(num_gpus=8, num_procs=8)),
    # https://aicarrier.feishu.cn/wiki/F7dFwgm1gixW49kWzg2cVKgzn4f / SFT-v0.0.2
    dict(abbr='ChatPJLM-v0.2.1rc2',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0429/sft_1006_instruct_v3/1799/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc2,
        max_out_len=100,max_seq_len=2048,batch_size=8,run_cfg=dict(num_gpus=8, num_procs=8)),
    # https://aicarrier.feishu.cn/wiki/IlvrwXOCPiVCPAknYpJcnhjpn1f / SFT-v0.1.0
    dict(abbr='ChatPJLM-v0.2.1',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0426/sft_1006_mixture0425/2099/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc1,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8, num_procs=8)),
]
