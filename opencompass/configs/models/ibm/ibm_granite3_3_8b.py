from opencompass.models.ibm_cloud_api import IBMGranite

models = [
    dict(
        type = IBMGranite,
        abbr = 'granite-3-3-8b-api',
        path = 'ibm/granite-3-3-8b-instruct',
        endpoint = 'https://us-south.ml.cloud.ibm.com/ml/v1/text/generation',
        project_id = 'f7b4594c-34b5-4d89-9c5b-95f4a287753a',
        key='ENV',  
        max_out_len = 1024,
        query_per_second = 2,
        run_cfg=dict(num_gpus=0),

    )
]