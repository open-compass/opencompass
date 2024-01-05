from opencompass.models import LLMv2
from mmengine.config import read_base

with read_base():
    from .._meta_templates import _template_0_1_0_ChatPJLM_v0_2_2rc5

models = [
    dict(abbr='ChatPJLM-v0.2.2rc51-Exam-v0.1.5',
        type=LLMv2, model_type='converted', path='opennlplab_hdd_new:s3://opennlplab_hdd_new/llm_it/0527/1006_sft_v022_with_exam_data/1599/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_2rc5,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8, num_nprocs=8)),
]
