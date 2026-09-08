general365_summary_groups = [
    dict(
        name='General365',
        subsets=['General365-mathverify', 'General365-text'],
        # General365_Public contains 484 math samples and 236 text samples.
        # Weighting by sample count reproduces the official micro-average over
        # all 720 samples instead of averaging the two subset accuracies.
        weights={
            'General365-mathverify': 484,
            'General365-text': 236,
        },
    )
]
