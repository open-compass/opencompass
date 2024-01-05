from mmengine.config import read_base

from opencompass.models.internal import LLMv2

with read_base():
    from .._meta_templates import _template_0_1_0_ChatPJLM_v0_2_2rc5

models = [
    dict(abbr='ChatPJLM-v0.2.0-Exam-v0.1.5',
        type=LLMv2, model_type='converted', path='opennlplab_hdd_new:s3://opennlplab_hdd_new/llm_it/0526/sft_1006_kaoshi_v022rc5/1599/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_2rc5,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8, num_procs=8)),
]
