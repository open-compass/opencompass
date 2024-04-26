from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='dbrx-instruct-hf',
        path='databricks/dbrx-instruct',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=8),
    )
]
