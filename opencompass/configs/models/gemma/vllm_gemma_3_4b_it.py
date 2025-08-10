from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='gemma-3-4b-it-vllm',
        path='google/gemma-3-4b-it',
        model_kwargs=dict(tensor_parallel_size=2, 
                          # for long context
                          rope_scaling={'factor': 8.0, 'rope_type': 'linear'}),
        max_seq_len=140000,
        max_out_len=4096,
        batch_size=1,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=2),
    )
]
