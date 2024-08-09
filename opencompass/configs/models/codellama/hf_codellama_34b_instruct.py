from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='CodeLlama-34b-Instruct',
        path='codellama/CodeLlama-34b-Instruct-hf',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]
