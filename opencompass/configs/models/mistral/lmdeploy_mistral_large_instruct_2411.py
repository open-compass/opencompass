from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='mixtral-large-instruct-2411-turbomind',
        path='mistralai/Mistral-Large-Instruct-2411',
        engine_config=dict(
            session_len=32768,
            max_batch_size=16,
            tp=4,
            cache_max_entry_count=0.7,
        ),
        gen_config=dict(
            top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=4096
        ),
        max_seq_len=32768,
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
