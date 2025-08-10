longbench_summary_groups = [
    {'name': 'longbench_single-document-qa', 'subsets': ['LongBench_narrativeqa', 'LongBench_qasper', 'LongBench_multifieldqa_en', 'LongBench_multifieldqa_zh']},
    {'name': 'longbench_multi-document-qa', 'subsets': ['LongBench_hotpotqa', 'LongBench_2wikimqa', 'LongBench_musique', 'LongBench_dureader']},
    {'name': 'longbench_summarization', 'subsets': ['LongBench_gov_report', 'LongBench_qmsum', 'LongBench_multi_news', 'LongBench_vcsum']},
    {'name': 'longbench_few-shot-learning', 'subsets': ['LongBench_trec', 'LongBench_triviaqa', 'LongBench_samsum', 'LongBench_lsht']},
    {'name': 'longbench_synthetic-tasks', 'subsets': ['LongBench_passage_count', 'LongBench_passage_retrieval_en', 'LongBench_passage_retrieval_zh']},
    {'name': 'longbench_code-completion', 'subsets': ['LongBench_lcc', 'LongBench_repobench-p']},

    # code tasks are included in both longbench_zh and longbench_en
    {'name': 'longbench_zh', 'subsets': ['LongBench_multifieldqa_zh', 'LongBench_dureader', 'LongBench_vcsum',
                                         'LongBench_lsht', 'LongBench_passage_retrieval_zh',
                                         'LongBench_lcc', 'LongBench_repobench-p']},
    {'name': 'longbench_en', 'subsets': [
        'LongBench_narrativeqa', 'LongBench_qasper', 'LongBench_multifieldqa_en',
        'LongBench_hotpotqa', 'LongBench_2wikimqa', 'LongBench_musique',
        'LongBench_gov_report', 'LongBench_qmsum', 'LongBench_multi_news',
        'LongBench_trec', 'LongBench_triviaqa', 'LongBench_samsum',
        'LongBench_passage_count', 'LongBench_passage_retrieval_en',
        'LongBench_lcc', 'LongBench_repobench-p'
    ]},
    {'name': 'longbench', 'subsets': ['longbench_single-document-qa', 'longbench_multi-document-qa', 'longbench_summarization', 'longbench_few-shot-learning', 'longbench_synthetic-tasks', 'longbench_code-completion']},
]
