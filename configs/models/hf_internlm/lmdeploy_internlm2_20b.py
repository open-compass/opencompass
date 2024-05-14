from opencompass.models.turbomind import TurboMindModel


models = [
    dict(
        type=TurboMindModel,
        abbr='internlm2-20b-turbomind',
        path='internlm/internlm2-20b',
        engine_config=dict(
            session_len=32768,
            max_batch_size=32,
            model_name='internlm2-20b',
            tp=2,
        ),
        gen_config=dict(
            top_k=1,
            top_p=0.8,
            temperature=1.0,
            max_new_tokens=2000,
        ),
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=32,
        concurrency=8,
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
