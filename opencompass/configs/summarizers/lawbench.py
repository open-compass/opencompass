from mmengine.config import read_base

with read_base():
    from .groups.lawbench import lawbench_summary_groups

summarizer = dict(
    dataset_abbrs = [
        '--------- 0-shot ---------', # category
        'lawbench-0-shot',
        'lawbench-1-1-article_recitation-0-shot',
        'lawbench-1-2-knowledge_question_answering-0-shot',
        'lawbench-2-1-document_proofreading-0-shot',
        'lawbench-2-2-dispute_focus_identification-0-shot',
        'lawbench-2-3-marital_disputes_identification-0-shot',
        'lawbench-2-4-issue_topic_identification-0-shot',
        'lawbench-2-5-reading_comprehension-0-shot',
        'lawbench-2-6-named_entity_recognition-0-shot',
        'lawbench-2-7-opinion_summarization-0-shot',
        'lawbench-2-8-argument_mining-0-shot',
        'lawbench-2-9-event_detection-0-shot',
        'lawbench-2-10-trigger_word_extraction-0-shot',
        'lawbench-3-1-fact_based_article_prediction-0-shot',
        'lawbench-3-2-scene_based_article_prediction-0-shot',
        'lawbench-3-3-charge_prediction-0-shot',
        'lawbench-3-4-prison_term_prediction_wo_article-0-shot',
        'lawbench-3-5-prison_term_prediction_w_article-0-shot',
        'lawbench-3-6-case_analysis-0-shot',
        'lawbench-3-7-criminal_damages_calculation-0-shot',
        'lawbench-3-8-consultation-0-shot',
        '--------- 1-shot ---------', # category
        'lawbench-1-shot',
        'lawbench-1-1-article_recitation-1-shot',
        'lawbench-1-2-knowledge_question_answering-1-shot',
        'lawbench-2-1-document_proofreading-1-shot',
        'lawbench-2-2-dispute_focus_identification-1-shot',
        'lawbench-2-3-marital_disputes_identification-1-shot',
        'lawbench-2-4-issue_topic_identification-1-shot',
        'lawbench-2-5-reading_comprehension-1-shot',
        'lawbench-2-6-named_entity_recognition-1-shot',
        'lawbench-2-7-opinion_summarization-1-shot',
        'lawbench-2-8-argument_mining-1-shot',
        'lawbench-2-9-event_detection-1-shot',
        'lawbench-2-10-trigger_word_extraction-1-shot',
        'lawbench-3-1-fact_based_article_prediction-1-shot',
        'lawbench-3-2-scene_based_article_prediction-1-shot',
        'lawbench-3-3-charge_prediction-1-shot',
        'lawbench-3-4-prison_term_prediction_wo_article-1-shot',
        'lawbench-3-5-prison_term_prediction_w_article-1-shot',
        'lawbench-3-6-case_analysis-1-shot',
        'lawbench-3-7-criminal_damages_calculation-1-shot',
        'lawbench-3-8-consultation-1-shot',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore'),
)
