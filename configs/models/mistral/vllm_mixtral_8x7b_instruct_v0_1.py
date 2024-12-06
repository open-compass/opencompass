from opencompass.models import VLLMwithChatTemplate


max_seq_len = 2048

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='mixtral-8x7b-instruct-v0.1-vllm',
        path='mistralai/Mixtral-8x7B-Instruct-v0.1',
        # more vllm model_kwargs: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        model_kwargs=dict(tensor_parallel_size=2, max_model_len=max_seq_len),
        max_out_len=256,
        max_seq_len=max_seq_len,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=2),
    )
]
