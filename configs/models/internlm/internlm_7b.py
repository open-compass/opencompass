from opencompass.models import InternLM


models = [
    dict(
        type=InternLM,
        path='./internData/',
        tokenizer_path='./internData/V7.model',
        model_config='./internData/model_config.py',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1))
]
