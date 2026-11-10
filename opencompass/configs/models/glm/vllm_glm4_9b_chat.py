from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='glm-4-9b-chat-vllm',
        path='THUDM/glm-4-9b-chat',
        max_out_len=1024,
        batch_size=16,
        model_kwargs=dict(tensor_parallel_size=1),
        run_cfg=dict(num_gpus=1),
        stop_words=['<|endoftext|>', '<|user|>', '<|observation|>'],
    )
]
