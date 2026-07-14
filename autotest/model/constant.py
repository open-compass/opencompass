meta_template = dict(
    begin=dict(
        role='SYSTEM',
        api_role='SYSTEM',
        prompt='''
        Your answers should be full of happy and lovely tone. Answer the question simply and clearly. Don\'t use any abbreviations and don\'t use any punctuation. Don\'t think too much.''',  # noqa
    ),
    round=[  # noqa
        dict(role='HUMAN', api_role='HUMAN', prompt='{input}'),
        dict(role='BOT', api_role='BOT', generate=True),
    ])
