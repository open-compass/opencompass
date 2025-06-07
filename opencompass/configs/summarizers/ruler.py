from mmengine.config import read_base

with read_base():
    from .groups.ruler import ruler_summary_groups


ruler_4k_summarizer = dict(
    dataset_abbrs=['ruler_4k'],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)

ruler_4k_summarizer = dict(
    dataset_abbrs=['ruler_4k'],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)
ruler_8k_summarizer = dict(
    dataset_abbrs=['ruler_8k'],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)
ruler_16k_summarizer = dict(
    dataset_abbrs=['ruler_16k'],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)
ruler_32k_summarizer = dict(
    dataset_abbrs=['ruler_32k'],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)
ruler_64k_summarizer = dict(
    dataset_abbrs=['ruler_64k'],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)
ruler_128k_summarizer = dict(
    dataset_abbrs=['ruler_128k'],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)
ruler_256k_summarizer = dict(
    dataset_abbrs=['ruler_256k'],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)
ruler_512k_summarizer = dict(
    dataset_abbrs=['ruler_512k'],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)

ruler_1m_summarizer = dict(
    dataset_abbrs=['ruler_1m'],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)

ruler_combined_summarizer = dict(
    dataset_abbrs=[
        'ruler_4k',
        'ruler_8k',
        'ruler_16k',
        'ruler_32k',
        'ruler_64k',
        'ruler_128k',
        'ruler_256k',
        'ruler_512k',
        'ruler_1m',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)
