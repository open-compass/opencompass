from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internvl2_5-38b-turbomind',
        path='OpenGVLab/InternVL2_5-38B',
        engine_config=dict(session_len=8192, max_batch_size=8, tp=4),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
        max_seq_len=8192,
        max_out_len=8192,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
