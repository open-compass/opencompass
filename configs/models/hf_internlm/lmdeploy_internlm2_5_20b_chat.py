from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internlm2_5-20b-chat-turbomind',
        path='internlm/internlm2_5-20b-chat',
        engine_config=dict(session_len=8192, max_batch_size=16, tp=2),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=2048),
        max_seq_len=8192,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=2),
    )
]
