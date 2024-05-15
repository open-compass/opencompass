from opencompass.models.vllm import VLLM


_meta_template = dict(
    begin='<s>',
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>\n', generate=True),
    ],
)

models = [
    dict(
        type=VLLM,
        abbr='internlm2-chat-7b-vllm',
        path='internlm/internlm2-chat-7b',
        model_kwargs=dict(tensor_parallel_size=1),
        meta_template=_meta_template,
        generation_kwargs=dict(
            top_k=1,
            temperature=0,
            stop_token_ids=[2, 92542],
        ),
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=32768,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
