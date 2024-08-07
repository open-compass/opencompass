from opencompass.models import VLLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|user|>\n', end='</s>'),
        dict(role='BOT', begin='<|assistant|>\n', end='</s>', generate=True),
    ],
)

models = [
    dict(
        type=VLLM,
        abbr='zephyr-7b-beta-vllm',
        path='HuggingFaceH4/zephyr-7b-beta',
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        stop_words=['</s>'],
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
