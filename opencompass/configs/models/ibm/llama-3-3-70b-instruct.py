from opencompass.models import IBMGranite

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='user'),
        dict(role='BOT', api_role='assistant', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='system')],
)

models = [
    dict(
        abbr='llama-3-3-70b-instruct',
        type=IBMGranite,
        path='meta-llama/llama-3-3-70b-instruct',
        api_key='ENV',
        project_id='ENV',
        region='us-south',
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=1024,
        max_seq_len=8192,
        batch_size=1,
        generation_kwargs=dict(
            decoding_method='sample',
            temperature=0.7,
            top_p=0.9,
        ),
    ),
]