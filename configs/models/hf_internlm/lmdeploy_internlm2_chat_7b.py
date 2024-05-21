from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internlm2-chat-7b-turbomind',
        path='internlm/internlm2-chat-7b',
        engine_config=dict(
            max_batch_size=16,
            tp=1,
        ),
        gen_config=dict(
            top_k=1,
            temperature=1e-6,
            top_p=0.9,
        ),
        max_seq_len=2048,
        max_out_len=1024,
        batch_size=32768,
        run_cfg=dict(num_gpus=1),
        stop_words=['</s>', '<|im_end|>'],
    )
]
