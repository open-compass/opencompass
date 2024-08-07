from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen-7b-chat-hf',
        path='Qwen/Qwen-7B-Chat',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|im_end|>', '<|im_start|>'],
    )
]
