from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='deepseek-67b-chat-vllm',
        path='deepseek-ai/deepseek-llm-67b-chat',
        max_out_len=1024,
        batch_size=16,
        model_kwargs=dict(tensor_parallel_size=4),
        run_cfg=dict(num_gpus=4),
    )
]
