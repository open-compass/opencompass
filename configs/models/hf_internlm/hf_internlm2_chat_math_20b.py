from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2-chat-math-20b-hf',
        path='internlm/internlm2-math-20b',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
        stop_words=['</s>', '<|im_end|>'],
    )
]
