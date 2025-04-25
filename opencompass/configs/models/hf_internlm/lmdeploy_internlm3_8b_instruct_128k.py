from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internlm3-8b-instruct-turbomind',
        path='internlm/internlm3-8b-instruct',
        engine_config=dict(session_len=142000, max_batch_size=1, tp=2,
                           # for long context
                           rope_scaling_factor=6.0),
        gen_config=dict(
            top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=8192
        ),
        max_seq_len=142000,
        max_out_len=8192,
        batch_size=1,
        run_cfg=dict(num_gpus=2),
    )
]
