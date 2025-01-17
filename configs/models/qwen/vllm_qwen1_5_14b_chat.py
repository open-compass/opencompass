from opencompass.models import VLLMwithChatTemplate

max_seq_len = 2048

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen1.5-14b-chat-vllm',
        path='Qwen/Qwen1.5-14B-Chat',
        # more vllm model_kwargs: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        model_kwargs=dict(tensor_parallel_size=2, max_model_len=max_seq_len),
        max_out_len=1024,
        max_seq_len=max_seq_len,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1),
    )
]
