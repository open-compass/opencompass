from opencompass.models.internal import LLM

models = [
    # PJLM-0.2 / 1006-15760
    dict(abbr="PJLM-0.2",
        type=LLM, path='model_weights:s3://model_weights/0331/1006/15760',
        tokenizer_path='/mnt/petrelfs/share_data/zhengmiao/llamav4.model', tokenizer_type='v4',
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8, num_procs=8)),
]
