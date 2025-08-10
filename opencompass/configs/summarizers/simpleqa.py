summarizer = dict(
    dataset_abbrs=[
        ['simpleqa', 'accuracy_given_attempted'],
        ['simpleqa', 'f1'],
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
