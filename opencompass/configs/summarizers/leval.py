summarizer = dict(
    dataset_abbrs = [
        '--------- LEval Exact Match (Acc) ---------', # category
        'LEval_coursera',
        'LEval_gsm100',
        'LEval_quality',
        'LEval_tpo',
        'LEval_topic_retrieval',
        '--------- LEval Gen (ROUGE) ---------', # category
        'LEval_financialqa',
        'LEval_gov_report_summ',
        'LEval_legal_contract_qa',
        'LEval_meeting_summ',
        'LEval_multidocqa',
        'LEval_narrativeqa',
        'LEval_nq',
        'LEval_news_summ',
        'LEval_paper_assistant',
        'LEval_patent_summ',
        'LEval_review_summ',
        'LEval_scientificqa',
        'LEval_tvshow_summ'
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
