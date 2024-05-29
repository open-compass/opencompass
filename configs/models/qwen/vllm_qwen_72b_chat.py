from opencompass.models import VLLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role='BOT', begin='\n<|im_start|>assistant\n', end='<|im_end|>', generate=True),
    ],
)
max_seq_len = 2048

models = [
    dict(
        type=VLLM,
        abbr='qwen-72b-chat-vllm',
        path='Qwen/Qwen-72B-Chat',
        # more vllm model_kwargs: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        model_kwargs=dict(tensor_parallel_size=4, max_model_len=max_seq_len),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=max_seq_len,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        stop_words=['<|im_end|>'],
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
