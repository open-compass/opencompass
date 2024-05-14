from opencompass.models.turbomind import TurboMindModel


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>\n', generate=True),
    ],
)

models = [
    dict(
        type=TurboMindModel,
        abbr='internlm2-chat-20b-turbomind',
        path='internlm/internlm2-chat-20b',
        meta_template=_meta_template,
        engine_config=dict(
            session_len=32768,
            max_batch_size=32,
            model_name='internlm2-chat-20b',
            tp=2,
            stop_words=[2, 92542],
        ),
        gen_config=dict(
            top_k=1,
            top_p=0.8,
            temperature=1.0,
            max_new_tokens=2000,
        ),
        max_out_len=2000,
        max_seq_len=32768,
        batch_size=32,
        concurrency=8,
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
