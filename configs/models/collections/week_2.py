from opencompass.models import LLM
from mmengine.config import read_base

with read_base():
    from .._meta_templates import _template_0_1_0_ChatPJLM_v0_2_1rc3, _template_0_1_0_ChatPJLM_v0_2_1rc1, _template_0_1_0_ChatPJLM_v0_2_2rc5

models = [
    # https://aicarrier.feishu.cn/wiki/Ak7xwPfagiGDBgkDDFNchy8wnMe
    dict(abbr='ChatPJLM-v0.2.1rc3',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0505/sft_1006_v003/1799/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc3,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8, num_procs=8)),
    # https://aicarrier.feishu.cn/wiki/YutBwecBEi1D1ckYAgmcr7dTnth
    dict(abbr='ChatPJLM-v0.2.2rc5',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0511/sft_1006_v022rc5/1599/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_2rc5,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8, num_procs=8)),
    # https://aicarrier.feishu.cn/wiki/X2xzwnn00iJXLkkUKLacO6RTn2f
    dict(abbr='ChatPJLM13B-v0.2.2rc2',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0510/sft_7132k_mixture0510/2399/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc1,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=2, num_procs=2)),
    # https://aicarrier.feishu.cn/wiki/TlfTwjhPziJ8ZakPVgfcVjKFnRd
    dict(abbr='ChatPJLM13B-v0.2.1rc4',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0506/mixture0425_nofirefly_belle/31999/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc1,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=2, num_procs=2)),
    # https://aicarrier.feishu.cn/wiki/Ya1Ww5v5firVgbk3JY5cPbVtn4g
    dict(abbr='ChatPJLM13B-v0.2.1rc5',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0425/sft_7132k_belle_2e_8196_1lr/9599/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc1,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=2, num_procs=2)),
    # https://aicarrier.feishu.cn/wiki/AslqwTA61i3F8Pkl2pXcwnMgnug
    dict(abbr='ChatPJLM13B-v0.2.1rc6',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0502/sft_7132k_sharegpt/1199/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc1,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=2, num_procs=2)),
    # https://aicarrier.feishu.cn/wiki/ZIwpwvLhpiBleTkwJnkcdpcsn5e
    dict(abbr='ChatPJLM13B-v0.2.2rc3',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0510/sft_7132k_zh_prompt512/119/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc1,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=2, num_procs=2)),
    # https://aicarrier.feishu.cn/wiki/GnjXw2jhBisRaHk4IkRcwvlynFh
    dict(abbr='ChatPJLM13B-v0.2.1rc1',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0502/sft_7132k_mixture0425/45999/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc1,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=2, num_procs=2)),
    # https://aicarrier.feishu.cn/wiki/QUArwQreVidEPAkIQgVcPYgMnFb
    dict(abbr='ChatPJLM13B-v0.2.2rc1',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0510/sft_7132k_firefly_no_belle_2e_8196_1lr/3799/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc1,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=2, num_procs=2)),
    # https://aicarrier.feishu.cn/wiki/X2xzwnn00iJXLkkUKLacO6RTn2f
    dict(abbr='ChatPJLM13B-v0.2.2rc2',
        type=LLM, path='opennlplab_hdd:s3://opennlplab_hdd/llm_it/0510/sft_7132k_mixture0510/2399/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc1,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=2, num_procs=2)),
]
