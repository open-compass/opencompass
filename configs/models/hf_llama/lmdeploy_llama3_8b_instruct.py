from opencompass.models import TurboMindModel

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|begin_of_text|>user<|end_header_id|>\n\n', end='<|eot_id|>'),
        dict(role='BOT', begin='<|begin_of_text|>assistant<|end_header_id|>\n\n', end='<|eot_id|>', generate=True),
    ],
)

models = [
    dict(
        type=TurboMindModel,
        abbr='llama-3-8b-instruct-lmdeploy',
        path='meta-llama/Meta-Llama-3-8B-Instruct',
        engine_config=dict(session_len=4096, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1, top_p=0.9, max_new_tokens=1024, stop_words=[128001, 128009]),
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=16,
        concurrency=16,
        meta_template=_meta_template,
        run_cfg=dict(num_gpus=1),
    )
]
