from opencompass.models import HuggingFaceAboveV433Chat

models = [
    dict(
        type=HuggingFaceAboveV433Chat,
        abbr='internlm2-chat-20b-sft-hf',
        path='internlm/internlm2-chat-20b-sft',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
        stop_words=['</s>', '<|im_end|>'],
    )
]
