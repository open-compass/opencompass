from opencompass.models import VLLM


_meta_template = dict(
    begin="<s>",
    round=[
        dict(role="HUMAN", begin='[INST]', end='[/INST]'),
        dict(role="BOT", begin="", end='</s>', generate=True),
    ],
    eos_token_id=2
)

models = [
    dict(
        type=VLLM,
        abbr='mixtral-8x7b-instruct-v0.1-vllm',
        path='mistralai/Mixtral-8x7B-Instruct-v0.1',
        model_kwargs=dict(tensor_parallel_size=2),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        end_str='</s>',
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
