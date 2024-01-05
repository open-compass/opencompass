from opencompass.models import LLM, LLama

models = [
    # LLama65B
    dict(abbr="LLama65B",
        type=LLama, path='/mnt/petrelfs/share_data/llm_llama/65B',
        tokenizer_path='/mnt/petrelfs/share_data/llm_llama/tokenizer.model', tokenizer_type='llama',
        max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=8, num_procs=8)),
    # PJLM-0.1 / 1006-10499
    dict(abbr="PJLM-0.1",
        type=LLM, path='/mnt/petrelfs/share_data/yanhang/weights/0331/1006/10499/',
        tokenizer_path='/mnt/petrelfs/share_data/zhengmiao/llamav4.model', tokenizer_type='v4',
        max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=8, num_procs=8)),
    # PJLM-0.2 / 1006-15760
    dict(abbr="PJLM-0.2",
        type=LLM, path='model_weights:s3://model_weights/0331/1006/15760',
        tokenizer_path='/mnt/petrelfs/share_data/zhengmiao/llamav4.model', tokenizer_type='v4',
        max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=8, num_procs=8)),
]
