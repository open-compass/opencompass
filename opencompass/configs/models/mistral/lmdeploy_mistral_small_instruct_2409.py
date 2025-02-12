from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='mistral-small-instruct-2409-turbomind',
        path='mistralai/Mistral-Small-Instruct-2409',
        engine_config=dict(session_len=32768, max_batch_size=16, tp=2),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=4096),
        max_seq_len=32768,
        max_out_len=4096,
        batch_size=16,
        run_cfg=dict(num_gpus=2),
    )
]
