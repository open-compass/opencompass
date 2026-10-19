from opencompass.models import VLLM


_meta_template = dict(
    begin='<s>',
    round=[
        dict(role='HUMAN', begin='Human: ', end='\n'),
        dict(role='BOT', begin='Assistant: ', end='</s>', generate=True),
    ],
)

models = [
    dict(
        abbr='orionstar-14b-longchat-vllm',
        type=VLLM,
        path='OrionStarAI/Orion-14B-LongChat',
        model_kwargs=dict(tensor_parallel_size=4),
        generation_kwargs=dict(temperature=0),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=4096,
        batch_size=32,
        run_cfg=dict(num_gpus=4, num_procs=1),
        stop_words=['<|endoftext|>'],
    )
]
