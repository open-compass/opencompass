from opencompass.models import VLLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>\n', generate=True),
    ],
)
max_seq_len = 2048

models = [
    dict(
        type=VLLM,
        abbr='qwen1.5-14b-chat-vllm',
        path='Qwen/Qwen1.5-14B-Chat',
        # more vllm model_kwargs: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        model_kwargs=dict(tensor_parallel_size=2, max_model_len=max_seq_len),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=max_seq_len,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        stop_words=['<|im_end|>'],
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
