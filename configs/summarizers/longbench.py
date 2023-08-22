summarizer = dict(
    dataset_abbrs = [
        '--------- LongBench Single-Document QA ---------', # category
        "LongBench_narrativeqa",
        'LongBench_qasper',
        'LongBench_multifieldqa_en',
        "LongBench_multifieldqa_zh",
        '--------- LongBench Multi-Document QA ---------', # category
        'LongBench_hotpotqa',
        'LongBench_2wikimqa',
        'LongBench_musique',
        'LongBench_dureader',
        '--------- LongBench Summarization ---------', # category
        'LongBench_gov_report',
        'LongBench_qmsum',
        'LongBench_vcsum',
        '--------- LongBench Few-shot Learning ---------', # category
        'LongBench_trec',
        'LongBench_nq',
        'LongBench_triviaqa',
        'LongBench_lsht',
        '--------- LongBench Code Completion ---------', # category
        'LongBench_lcc',
        'LongBench_repobench-p',
        '--------- LongBench Synthetic Tasks ---------', # category
        'LongBench_passage_retrieval_en',
        'LongBench_passage_count',
        'LongBench_passage_retrieval_zh',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith("_summary_groups")], []),
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore'),
)
