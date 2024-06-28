from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='llama-2-7b-chat-turbomind',
        path='meta-llama/Llama-2-7b-chat-hf',
        engine_config=dict(max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
        max_seq_len=4096,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]
