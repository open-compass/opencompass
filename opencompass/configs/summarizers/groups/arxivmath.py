arxivmath_summary_groups = [
    dict(
        name='arxivmath_08_to_10_avg@4',
        subsets=[
            ['arxivmath_0826', 'accuracy (4 runs average)'],
            ['arxivmath_0926', 'accuracy (4 runs average)'],
            ['arxivmath_1026', 'accuracy (4 runs average)'],
        ],
        # Weighting by problem count (57 / 53 / 47) gives the micro-average
        # accuracy over all 157 problems.
        weights={
            'arxivmath_0826': 57,
            'arxivmath_0926': 53,
            'arxivmath_1026': 47,
        },
    )
]
