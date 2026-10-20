from mmengine.config import read_base

with read_base():
    from .levalnaturalquestion.leval_naturalquestion_gen import LEval_nq_datasets
    from .levalnarrativeqa.leval_narrativeqa_gen import LEval_narrativeqa_datasets
    from .levalmultidocqa.leval_multidocqa_gen import LEval_multidocqa_datasets
    from .levalcoursera.leval_coursera_gen import LEval_coursera_datasets
    from .levaltpo.leval_tpo_gen import LEval_tpo_datasets
    from .levalquality.leval_quality_gen import LEval_quality_datasets
    from .levalgsm100.leval_gsm100_gen import LEval_gsm100_datasets
    from .levaltopicretrieval.leval_topic_retrieval_gen import LEval_tr_datasets
    from .levalfinancialqa.leval_financialqa_gen import LEval_financialqa_datasets
    from .levalgovreportsumm.leval_gov_report_summ_gen import LEval_govreport_summ_datasets
    from .levallegalcontractqa.leval_legalcontractqa_gen import LEval_legalqa_datasets
    from .levalmeetingsumm.leval_meetingsumm_gen import LEval_meetingsumm_datasets
    from .levalnewssumm.leval_newssumm_gen import LEval_newssumm_datasets
    from .levalpaperassistant.leval_paper_assistant_gen import LEval_ps_summ_datasets
    from .levalpatentsumm.leval_patent_summ_gen import LEval_patent_summ_datasets
    from .levaltvshowsumm.leval_tvshow_summ_gen import LEval_tvshow_summ_datasets
    from .levalscientificqa.leval_scientificqa_gen import LEval_scientificqa_datasets
    from .levalreviewsumm.leval_review_summ_gen import LEval_review_summ_datasets

leval_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
