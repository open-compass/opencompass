from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen-14b-chat-vllm',
        path='Qwen/Qwen-14B-Chat',
        # more vllm model_kwargs: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        model_kwargs=dict(tensor_parallel_size=1),
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=1024,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
