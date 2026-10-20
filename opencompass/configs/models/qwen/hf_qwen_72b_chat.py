from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen-72b-chat-hf',
        path='Qwen/Qwen-72B-Chat',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=8),
        stop_words=['<|im_end|>', '<|im_start|>'],
    )
]
