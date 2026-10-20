from opencompass.models import VLLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='USER: '),
        dict(role='BOT', begin=' ASSISTANT:', end='</s>', generate=True),
    ],
)

models = [
    dict(
        type=VLLM,
        abbr='vicuna-13b-v1.5-16k-vllm',
        path='lmsys/vicuna-13b-v1.5-16k',
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        stop_words=['</s>'],
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
