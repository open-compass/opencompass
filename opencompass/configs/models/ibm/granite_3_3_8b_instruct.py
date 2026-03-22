from opencompass.models.ibm_cloud_api import IBMGranite

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='user'),
        dict(role='BOT', api_role='assistant', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='system')],
)

models = [
    dict(
        abbr='granite-8b-code-instruct',
        type=IBMGranite,
        path='ibm/granite-8b-code-instruct',
        api_key='ENV',  # Obtained from $IBM_CLOUD_API_KEY when set to 'ENV'
        project_id='ENV',  # Obtained from $IBM_CLOUD_PROJECT_ID when set to 'ENV'
        region='us-south',
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=1,
        generation_kwargs=dict(
            decoding_method='sample',
            temperature=0.7,
            top_p=0.9,
        ),
    ),
]

