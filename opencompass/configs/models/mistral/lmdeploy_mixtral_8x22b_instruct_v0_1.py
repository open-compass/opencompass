from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='mixtral-8x22b-instruct-v0.1-turbomind',
        path='mistralai/Mixtral-8x22B-Instruct-v0.1',
        engine_config=dict(
            session_len=32768,
            max_batch_size=16,
            tp=8,
            cache_max_entry_count=0.7,
        ),
        gen_config=dict(
            top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=4096
        ),
        max_seq_len=32768,
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=8),
    )
]
