from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2.5-32b-instruct-vllm',
        path='Qwen/Qwen2.5-32B-Instruct',
        model_kwargs=dict(
            tensor_parallel_size=8,
            rope_scaling={
                'factor': 4.0,
                'original_max_position_embeddings': 32768,
                'rope_type': 'yarn'
            },
        ),
        max_out_len=4096,
        batch_size=1,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=8),
    )
]
