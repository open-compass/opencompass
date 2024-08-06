from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='yi-34b-chat-hf',
        path='01-ai/Yi-34B-Chat',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
        stop_words=['<|endoftext|>', '<|im_end|>'],
    )
]
