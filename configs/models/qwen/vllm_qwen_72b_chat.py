from opencompass.models import VLLM


_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="\n<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)

models = [
    dict(
        type=VLLM,
        abbr='qwen-72b-chat-vllm',
        path="Qwen/Qwen-72B-Chat",
        model_kwargs=dict(tensor_parallel_size=4),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        end_str='<|im_end|>',
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
