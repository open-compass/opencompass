from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='qwen1.5-110b-chat-turbomind',
        path='Qwen/Qwen1.5-110B-Chat',
        engine_config=dict(session_len=16834, max_batch_size=8, tp=4),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=4096),
        max_seq_len=16834,
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
        stop_words=['<|im_end|>', '<|im_start|>'],
    )
]
