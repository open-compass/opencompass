from mmengine.config import read_base

with read_base():
    from .longbench2wikimqa.longbench_2wikimqa_rawprompt_gen import LongBench_2wikimqa_datasets
    from .longbenchhotpotqa.longbench_hotpotqa_rawprompt_gen import LongBench_hotpotqa_datasets
    from .longbenchmusique.longbench_musique_rawprompt_gen import LongBench_musique_datasets
    from .longbenchmultifieldqa_en.longbench_multifieldqa_en_rawprompt_gen import LongBench_multifieldqa_en_datasets
    from .longbenchmultifieldqa_zh.longbench_multifieldqa_zh_rawprompt_gen import LongBench_multifieldqa_zh_datasets
    from .longbenchnarrativeqa.longbench_narrativeqa_rawprompt_gen import LongBench_narrativeqa_datasets
    from .longbenchqasper.longbench_qasper_rawprompt_gen import LongBench_qasper_datasets
    from .longbenchtriviaqa.longbench_triviaqa_rawprompt_gen import LongBench_triviaqa_datasets
    from .longbenchgov_report.longbench_gov_report_rawprompt_gen import LongBench_gov_report_datasets
    from .longbenchqmsum.longbench_qmsum_rawprompt_gen import LongBench_qmsum_datasets
    from .longbenchvcsum.longbench_vcsum_rawprompt_gen import LongBench_vcsum_datasets
    from .longbenchdureader.longbench_dureader_rawprompt_gen import LongBench_dureader_datasets
    from .longbenchlcc.longbench_lcc_rawprompt_gen import LongBench_lcc_datasets
    from .longbenchrepobench.longbench_repobench_rawprompt_gen import LongBench_repobench_datasets
    from .longbenchpassage_retrieval_en.longbench_passage_retrieval_en_rawprompt_gen import LongBench_passage_retrieval_en_datasets
    from .longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_rawprompt_gen import LongBench_passage_retrieval_zh_datasets
    from .longbenchpassage_count.longbench_passage_count_rawprompt_gen import LongBench_passage_count_datasets
    from .longbenchtrec.longbench_trec_rawprompt_gen import LongBench_trec_datasets
    from .longbenchlsht.longbench_lsht_rawprompt_gen import LongBench_lsht_datasets
    from .longbenchmulti_news.longbench_multi_news_rawprompt_gen import LongBench_multi_news_datasets
    from .longbenchsamsum.longbench_samsum_rawprompt_gen import LongBench_samsum_datasets

longbench_rawprompt_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
