from opencompass.models import LLM
from mmengine.config import read_base

with read_base():
    from .._meta_templates import _template_0_1_0_ChatPJLM_v0_2_2rc5

models = [
    # https://aicarrier.feishu.cn/wiki/YgxYwDu7eiuT9SkPp48c3F13n2g
    dict(abbr='ChatPJLM13B-v0.2.2rc5',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0510/sft_7132k_v022rc5/35999/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_2rc5,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=2, num_procs=2)),
]
