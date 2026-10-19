from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='deepseek-moe-16b-chat-vllm',
        path='deepseek-ai/deepseek-moe-16b-chat',
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.6),
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]
