from opencompass.models import AntFinixAPI

models = [
    dict(
        path='035A54D2-9A48-021A-8ED7-C6758F3344AF',
        key='',  # set your key here or in environment variable ANTFINIX_API_KEY
        url='https://fin-evaluator-gw.antgroup.com/api/v1/finEvaluator/evaluate',
        type=AntFinixAPI,
        max_out_len=32 * 1024,
        batch_size=1,
        generation_kwargs={
            'temperature': 1.0,
            'logprobs': 0,
            'top_p': 1.0,
            'top_k': -1,
            'n': 1,
        },
    ),
]
