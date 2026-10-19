from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internlm2_5-7b-chat-1m-turbomind',
        path='internlm/internlm2_5-7b-chat-1m',
        engine_config=dict(rope_scaling_factor=2.5, session_len=1048576, max_batch_size=1, cache_max_entry_count=0.7, tp=4), # 1M context length
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=2048),
        max_seq_len=1048576,
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=4),
    )
]
