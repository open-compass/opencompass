from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='chatglm3-6b-32k-vllm',
        path='THUDM/chatglm3-6b-32k',
        max_out_len=100,
        max_seq_len=4096,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
