from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-67b-chat-turbomind',
        path='deepseek-ai/deepseek-llm-67b-chat',
        engine_config=dict(max_batch_size=16, tp=4),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9),
        max_seq_len=7168,
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=4),
    )
]
