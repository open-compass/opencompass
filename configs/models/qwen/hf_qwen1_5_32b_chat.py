from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen1.5-32b-chat-hf',
        path='Qwen/Qwen1.5-32B-Chat',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
        stop_words=['<|im_end|>', '<|im_start|>'],
    )
]
