
_longeval_2k = ['classification_en_2k', 'lines_2k', 'qa_en_2k', 'qa_zh_2k', 'stackselect_2k', 'summarization_en_2k', 'textsort_2k']
_longeval_4k = ['classification_en_4k', 'lines_4k', 'qa_en_4k', 'qa_zh_4k', 'stackselect_4k', 'summarization_en_4k', 'textsort_4k']
_longeval_8k = ['classification_en_8k', 'lines_8k', 'qa_en_8k', 'qa_zh_8k', 'stackselect_8k', 'summarization_en_8k', 'textsort_8k']
_longeval_15k = ['classification_en_15k', 'lines_15k', 'qa_en_15k', 'qa_zh_15k', 'stackselect_15k', 'summarization_en_15k', 'textsort_15k']
_longeval_30k = ['classification_en_30k', 'lines_30k', 'qa_en_30k', 'qa_zh_30k', 'stackselect_30k', 'summarization_en_30k', 'textsort_30k']

longeval_summary_groups = [
    {'name': 'longeval_v2_2k', 'subsets': _longeval_2k},
    {'name': 'longeval_v2_4k', 'subsets': _longeval_4k},
    {'name': 'longeval_v2_8k', 'subsets': _longeval_8k},
    {'name': 'longeval_v2_15k', 'subsets': _longeval_15k},
    {'name': 'longeval_v2_30k', 'subsets': _longeval_30k},
    {'name': 'longeval_v2', 'subsets': _longeval_2k + _longeval_4k + _longeval_8k + _longeval_15k + _longeval_30k}
]
summarizer = dict(
    dataset_abbrs = [
        'longeval_v2',
        'longeval_v2_2k',
        'longeval_v2_4k',
        'longeval_v2_8k',
        'longeval_v2_15k',
        'longeval_v2_30k',
        'classification_en_2k',
        'classification_en_4k',
        'classification_en_8k',
        'classification_en_15k',
        'classification_en_30k',
        'lines_2k',
        'lines_4k',
        'lines_8k',
        'lines_15k',
        'lines_30k',
        'qa_en_2k',
        'qa_en_4k',
        'qa_en_8k',
        'qa_en_15k',
        'qa_en_30k',
        'qa_zh_2k',
        'qa_zh_4k',
        'qa_zh_8k',
        'qa_zh_15k',
        'qa_zh_30k',
        'stackselect_2k',
        'stackselect_4k',
        'stackselect_8k',
        'stackselect_15k',
        'stackselect_30k',
        'summarization_en_2k',
        'summarization_en_4k',
        'summarization_en_8k',
        'summarization_en_15k',
        'summarization_en_30k',
        'textsort_2k',
        'textsort_4k',
        'textsort_8k',
        'textsort_15k',
        'textsort_30k',
    ],
    summary_groups=longeval_summary_groups,
)
