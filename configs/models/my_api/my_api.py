from opencompass.models.my_api import MyAPIModel
api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)
models = [
    dict(
        type=MyAPIModel,
        abbr='my_api',
        path='my_api',
        url='https://api-opencompass.jd.com/testing',
        api_key = 'w8QA7LSXQG1q9Tc1A0X3P8PWXMkmyuPSCPtRSCg9NtM95dBlpO',
        meta_template=api_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]