from opencompass.models import LLM, LLama, LLMv2
from mmengine.config import read_base

with read_base():
    from .._meta_templates import _template_0_1_0_ChatPJLM_v0_2_2rc5

models = [
    dict(abbr="LLama65B",
        type=LLama, path='/mnt/petrelfs/share_data/llm_llama/65B',
        tokenizer_path='/mnt/petrelfs/share_data/llm_llama/tokenizer.model', tokenizer_type='llama',
        max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=8, num_procs=8)),
    dict(abbr="PJLM-0.1",
        type=LLM, path='model_weights:s3://model_weights/0331/1006/10499',
        tokenizer_path='/mnt/petrelfs/share_data/zhengmiao/llamav4.model', tokenizer_type='v4',
        max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=8, num_procs=8)),
    dict(abbr="PJLM-0.2",
        type=LLM, path='model_weights:s3://model_weights/0331/1006/15760',
        tokenizer_path='/mnt/petrelfs/share_data/zhengmiao/llamav4.model', tokenizer_type='v4',
        max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=8, num_procs=8)),
    dict(abbr='PJLM-v0.2.0-Exam-v0.1.5',
        type=LLMv2, model_type='converted', path='model_weights:s3://model_weights/0331/1006_pr/5499/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8, num_procs=8)),
    dict(abbr='ChatPJLM-v0.2.0-Exam-v0.1.5',
        type=LLMv2, model_type='converted', path='opennlplab_hdd_new:s3://opennlplab_hdd_new/llm_it/0526/sft_1006_kaoshi_v022rc5/1599/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_2rc5,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8, num_procs=8)),
    dict(abbr='ChatPJLM-v0.2.2rc51-Exam-v0.1.5',
        type=LLMv2, model_type='converted', path='opennlplab_hdd_new:s3://opennlplab_hdd_new/llm_it/0527/1006_sft_v022_with_exam_data/1599/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_2rc5,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8, num_procs=8)),
]
