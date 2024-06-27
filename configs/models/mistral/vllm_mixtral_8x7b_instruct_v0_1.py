from opencompass.models import VLLM


_meta_template = dict(
    begin='<s>',
    round=[
        dict(role='HUMAN', begin='[INST]', end='[/INST]'),
        dict(role='BOT', begin='', end='</s>', generate=True),
    ],
)
max_seq_len = 2048

models = [
    dict(
        type=VLLM,
        abbr='mixtral-8x7b-instruct-v0.1-vllm',
        path='mistralai/Mixtral-8x7B-Instruct-v0.1',
        # more vllm model_kwargs: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        model_kwargs=dict(tensor_parallel_size=2, max_model_len=max_seq_len),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=max_seq_len,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        stop_words=['</s>'],
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
