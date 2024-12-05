from opencompass.models import TurboMindModel

models = [
    dict(
        type=TurboMindModel,
        abbr='gemma-2-9b-turbomind',
        path='google/gemma-2-9b',
        engine_config=dict(session_len=16384, max_batch_size=16, tp=1),
        gen_config=dict(
            top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=4096
        ),
        max_seq_len=16384,
        max_out_len=4096,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]
