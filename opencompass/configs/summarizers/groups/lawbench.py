names = [
    ['1-1', 'article_recitation'],
    ['1-2', 'knowledge_question_answering'],
    ['2-1', 'document_proofreading'],
    ['2-2', 'dispute_focus_identification'],
    ['2-3', 'marital_disputes_identification'],
    ['2-4', 'issue_topic_identification'],
    ['2-5', 'reading_comprehension'],
    ['2-6', 'named_entity_recognition'],
    ['2-7', 'opinion_summarization'],
    ['2-8', 'argument_mining'],
    ['2-9', 'event_detection'],
    ['2-10', 'trigger_word_extraction'],
    ['3-1', 'fact_based_article_prediction'],
    ['3-2', 'scene_based_article_prediction'],
    ['3-3', 'charge_prediction'],
    ['3-4', 'prison_term_prediction_wo_article'],
    ['3-5', 'prison_term_prediction_w_article'],
    ['3-6', 'case_analysis'],
    ['3-7', 'criminal_damages_calculation'],
    ['3-8', 'consultation'],
]

lawbench_summary_groups = []

_lawbench_0_shot = ['lawbench-' + index + '-' + name + '-0-shot' for index, name in names]
lawbench_summary_groups.append({'name': 'lawbench-0-shot', 'subsets': _lawbench_0_shot})
_lawbench_1_shot = ['lawbench-' + index + '-' + name + '-1-shot' for index, name in names]
lawbench_summary_groups.append({'name': 'lawbench-1-shot', 'subsets': _lawbench_1_shot})
