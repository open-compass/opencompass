from opencompass.models import TurboMindModel

models = [
    dict(
        type=TurboMindModel,
        abbr='glm-4-9b-turbomind',
        path='THUDM/glm-4-9b',
        engine_config=dict(max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=2048),
        max_seq_len=8192,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]
