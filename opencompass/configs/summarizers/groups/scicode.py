from copy import deepcopy
from opencompass.summarizers import DefaultSummarizer


scicode_summary_groups = [
    {
        'name': 'SciCode',
        'type': DefaultSummarizer,
        'subsets': [
            ['SciCode', 'accuracy'],
            ['SciCode', 'sub_accuracy'],
        ]
    },
    {
        'name': 'SciCode_with_background',
        'type': DefaultSummarizer,
        'subsets': [
            ['SciCode_with_background', 'accuracy'],
            ['SciCode_with_background', 'sub_accuracy'],
        ]
    },
    {
        'name': 'SciCode_wo_background',
        'type': DefaultSummarizer,
        'subsets': [
            ['SciCode_wo_background', 'accuracy'],
            ['SciCode_wo_background', 'sub_accuracy'],
        ]
    }
]
