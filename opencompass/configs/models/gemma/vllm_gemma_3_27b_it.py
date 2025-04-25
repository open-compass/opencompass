from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='gemma-3-27b-it-vllm',
        path='google/gemma-3-27b-it',
        model_kwargs=dict(tensor_parallel_size=4,
                          # for long context
                          rope_scaling={'factor': 8.0, 'rope_type': 'linear'}),
        max_out_len=4096,
        batch_size=1,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=4),
    )   
]
