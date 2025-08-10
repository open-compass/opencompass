from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen-14b-chat-hf',
        path='Qwen/Qwen-14B-Chat',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
        stop_words=['<|im_end|>', '<|im_start|>'],
    )
]
