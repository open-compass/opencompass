from opencompass.models.internal import LLama

models = [
    # LLama65B
    dict(abbr="LLama65B",
        type=LLama, path='/mnt/petrelfs/share_data/llm_llama/65B',
        tokenizer_path='/mnt/petrelfs/share_data/llm_llama/tokenizer.model', tokenizer_type='llama',
        max_out_len=100, max_seq_len=2048, batch_size=16, run_cfg=dict(num_gpus=8, num_procs=8))
]
