from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3_1-70b-instruct-hf',
        path='meta-llama/Meta-Llama-3.1-70B-Instruct',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]
