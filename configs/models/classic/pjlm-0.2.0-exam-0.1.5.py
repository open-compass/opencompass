from opencompass.models.internal import LLMv2

models = [
    dict(abbr='PJLM-v0.2.0-Exam-v0.1.5',
        type=LLMv2, model_type='converted', path='model_weights:s3://model_weights/0331/1006_pr/5499/',
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model', tokenizer_type='v4',
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8, num_procs=8)),
]
