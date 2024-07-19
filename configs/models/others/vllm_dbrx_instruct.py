from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='dbrx-instruct-vllm',
        path='databricks/dbrx-instruct',
        model_kwargs=dict(tensor_parallel_size=8),
        max_out_len=1024,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=8),
    )
]
