
from mmengine.config import read_base

with read_base():
    from .groups.legacy.cibench import cibench_summary_groups
    from .groups.plugineval import plugineval_summary_groups


compassbench_v1_language_names = [
    # ['information_retrieval_en', 'score'],
    # ['information_retrieval_zh', 'score'],
    ['intention_recognition_en_circular', 'acc_origin'],
    ['intention_recognition_en_circular', 'perf_circular'],
    ['intention_recognition_zh_circular', 'acc_origin'],
    ['intention_recognition_zh_circular', 'perf_circular'],
    ['sentiment_analysis_en_circular', 'acc_origin'],
    ['sentiment_analysis_en_circular', 'perf_circular'],
    ['sentiment_analysis_zh_circular', 'acc_origin'],
    ['sentiment_analysis_zh_circular', 'perf_circular'],
    ['translation', 'score'],
    ['content_critic_en_circular', 'acc_origin'],
    ['content_critic_en_circular', 'perf_circular'],
    ['content_critic_zh_circular', 'acc_origin'],
    ['content_critic_zh_circular', 'perf_circular'],
    ['content_summarization_en', 'rouge1'],
    ['content_summarization_zh', 'rouge1'],
    ['traditional_cultural_understanding_zh_circular', 'acc_origin'],
    ['traditional_cultural_understanding_zh_circular', 'perf_circular'],
    ['chinese_semantic_understanding_zh_circular', 'acc_origin'],
    ['chinese_semantic_understanding_zh_circular', 'perf_circular'],
]

compassbench_v1_language_summary_groups = [
    {'name': 'language_zh_acc_1_and_non_mcq', 'subsets': [[name, metric] for name, metric in compassbench_v1_language_names if '_zh' in name and metric != 'perf_circular']},
    {'name': 'language_en_acc_1_and_non_mcq', 'subsets': [[name, metric] for name, metric in compassbench_v1_language_names if '_en' in name and metric != 'perf_circular']},
    {'name': 'language_acc_1_and_non_mcq', 'subsets': ['language_zh_acc_1_and_non_mcq', 'language_en_acc_1_and_non_mcq']},

    {'name': 'language_zh_perf_4_and_non_mcq', 'subsets': [[name, metric] for name, metric in compassbench_v1_language_names if '_zh' in name and metric != 'acc_origin']},
    {'name': 'language_en_perf_4_and_non_mcq', 'subsets': [[name, metric] for name, metric in compassbench_v1_language_names if '_en' in name and metric != 'acc_origin']},
    {'name': 'language_perf_4_and_non_mcq', 'subsets': ['language_zh_perf_4_and_non_mcq', 'language_en_perf_4_and_non_mcq']},
]

# This summarizer is used for `./datasets/compassbench_v1_knowledge/compassbench_v1_knowledge_gen`
compassbench_v1_knowledge_names = [
    'compassbench_v1_knowledge-common_knowledge-single_choice_cn_circular',
    'compassbench_v1_knowledge-humanity-single_choice_cn_circular',
    'compassbench_v1_knowledge-natural_science-single_choice_cn_circular',
    'compassbench_v1_knowledge-social_science-single_choice_cn_circular',
]

compassbench_v1_knowledge_summary_groups = [
    {'name': 'knowledge_cn', 'subsets': compassbench_v1_knowledge_names},
    {'name': 'knowledge_acc_1_and_cloze', 'subsets': [['knowledge_cn', 'acc_1'], ['compassbench_v1_knowledge-mixed-cloze_en', 'score']]},
    {'name': 'knowledge_perf_4_and_cloze', 'subsets': [['knowledge_cn', 'perf_4'], ['compassbench_v1_knowledge-mixed-cloze_en', 'score']]},
]

compassbench_v1_reason_summary_groups = [
    {'name': 'reasonbench_cn_abductive_circular', 'subsets': ['reasonbench_cn_abductive_alphanlg_translated_circular']},
    {'name': 'reasonbench_en_abductive_circular', 'subsets': ['reasonbench_en_abductive_alphanlg_circular']},
    {'name': 'reasonbench_cn_deductive_circular', 'subsets': ['reasonbench_cn_deductive_bbh3obj_translated_circular', 'reasonbench_cn_deductive_logiqa_zh_circular']},
    {'name': 'reasonbench_cn_inductive_circular', 'subsets': ['reasonbench_cn_inductive_deer_translated_circular', 'reasonbench_cn_inductive_selfgenerated_circular']},
    {'name': 'reasonbench_en_inductive_circular', 'subsets': ['reasonbench_en_inductive_deer_circular', 'reasonbench_en_inductive_selfgenerated_circular']},

    {'name': 'reasonbench_cn_circular', 'subsets': ['reasonbench_cn_commonsense_circular', 'reasonbench_cn_abductive_circular', 'reasonbench_cn_deductive_circular', 'reasonbench_cn_inductive_circular']},
    {'name': 'reasonbench_en_circular', 'subsets': ['reasonbench_en_commonsense_circular', 'reasonbench_en_abductive_circular', 'reasonbench_en_deductive_logiqa_zh_translated_circular', 'reasonbench_en_inductive_circular']},
    {'name': 'reasonbench', 'subsets': ['reasonbench_cn_circular', 'reasonbench_en_circular']},
]

compassbench_v1_math_summary_groups = [
    {'name': 'math_acc_1_and_fill_in_blank', 'subsets': [['compassbench_v1_math-high-single_choice_cn', 'acc_1'], ['compassbench_v1_math-high-single_choice_en', 'acc_1'], ['compassbench_v1_math-middle-single_choice_cn', 'acc_1'], ['compassbench_v1_math-middle-single_choice_en', 'acc_1'], ['compassbench_v1_math-primary-cloze_cn', 'accuracy'], ['compassbench_v1_math-primary-cloze_en', 'accuracy']]},
    {'name': 'math_perf_4_and_fill_in_blank', 'subsets': [['compassbench_v1_math-high-single_choice_cn', 'perf_4'], ['compassbench_v1_math-high-single_choice_en', 'perf_4'], ['compassbench_v1_math-middle-single_choice_cn', 'perf_4'], ['compassbench_v1_math-middle-single_choice_en', 'perf_4'], ['compassbench_v1_math-primary-cloze_cn', 'accuracy'], ['compassbench_v1_math-primary-cloze_en', 'accuracy']]},
    {'name': 'math_perf_4_and_fill_in_blank_cn', 'subsets': [['compassbench_v1_math-high-single_choice_cn', 'perf_4'], ['compassbench_v1_math-middle-single_choice_cn', 'perf_4'], ['compassbench_v1_math-primary-cloze_cn', 'accuracy']]},
    {'name': 'math_perf_4_and_fill_in_blank_en', 'subsets': [['compassbench_v1_math-high-single_choice_en', 'perf_4'], ['compassbench_v1_math-middle-single_choice_en', 'perf_4'], ['compassbench_v1_math-primary-cloze_en', 'accuracy']]},
]


code_passk_summary_groups = [
    # rename
    {'name': 'humaneval_pass@1(greedy)', 'subsets': [['openai_humaneval', 'humaneval_pass@1']]},
    {'name': 'humaneval_pass@10', 'subsets': [['openai_humaneval_passk', 'humaneval_pass@10']]},
    {'name': 'humaneval_pass@10', 'subsets': [['openai_humaneval_repeat10', 'humaneval_pass@10']]},
    {'name': 'humaneval_cn_pass@1(greedy)', 'subsets': [['openai_humaneval_cn', 'humaneval_pass@1']]},
    {'name': 'humaneval_cn_pass@10', 'subsets': [['openai_humaneval_cn_passk', 'humaneval_pass@10']]},
    {'name': 'humaneval_cn_pass@10', 'subsets': [['openai_humaneval_cn_repeat10', 'humaneval_pass@10']]},
    {'name': 'humaneval_plus_pass@1(greedy)', 'subsets': [['humaneval_plus', 'humaneval_plus_pass@1']]},
    {'name': 'humaneval_plus_pass@10', 'subsets': [['humaneval_plus_passk', 'humaneval_plus_pass@10']]},
    {'name': 'humaneval_plus_pass@10', 'subsets': [['humaneval_plus_repeat10', 'humaneval_plus_pass@10']]},
    {'name': 'mbpp_pass@1(greedy)', 'subsets': [['mbpp', 'score']]},
    {'name': 'mbpp_pass@10', 'subsets': [['mbpp_passk', 'pass@10']]},
    {'name': 'mbpp_pass@10', 'subsets': [['mbpp_repeat10', 'pass@10']]},
    {'name': 'mbpp_cn_pass@1(greedy)', 'subsets': [['mbpp_cn', 'score']]},
    {'name': 'mbpp_cn_pass@10', 'subsets': [['mbpp_cn_passk', 'pass@10']]},
    {'name': 'mbpp_cn_pass@10', 'subsets': [['mbpp_cn_repeat10', 'pass@10']]},
    {'name': 'sanitized_mbpp_pass@1(greedy)', 'subsets': [['sanitized_mbpp', 'score']]},
    {'name': 'sanitized_mbpp_pass@10', 'subsets': [['sanitized_mbpp_passk', 'pass@10']]},
    {'name': 'sanitized_mbpp_pass@10', 'subsets': [['sanitized_mbpp_repeat10', 'pass@10']]},
    # real add
    {'name': 'humanevalx', 'subsets': ['humanevalx-python', 'humanevalx-cpp', 'humanevalx-go', 'humanevalx-java', 'humanevalx-js']},
    {'name': 'code_cn', 'subsets': ['humaneval_cn_pass@1(greedy)', 'mbpp_cn_pass@1(greedy)']},
    {'name': 'code_en', 'subsets': ['humaneval_plus_pass@1(greedy)', 'sanitized_mbpp_pass@1(greedy)', 'humanevalx']},
    {'name': 'code', 'subsets': ['humaneval_cn_pass@1(greedy)', 'mbpp_cn_pass@1(greedy)', 'humaneval_plus_pass@1(greedy)', 'sanitized_mbpp_pass@1(greedy)', 'humanevalx']},
]

agent_summary_groups = [
    # dict(name='cibench_template', subsets=['cibench_template:executable', 'cibench_template:numeric_correct', 'cibench_template:text_score', 'cibench_template:vis_sim']),
    # dict(name='cibench_template_cn', subsets=['cibench_template_cn:executable', 'cibench_template_cn:numeric_correct', 'cibench_template_cn:text_score', 'cibench_template_cn:vis_sim']),

    dict(name='cibench_template', subsets=['cibench_template_wo_nltk:executable', 'cibench_template_wo_nltk:numeric_correct', 'cibench_template_wo_nltk:vis_sim']),
    dict(name='cibench_template_cn', subsets=['cibench_template_cn_wo_nltk:executable', 'cibench_template_cn_wo_nltk:numeric_correct', 'cibench_template_cn_wo_nltk:vis_sim']),

    dict(name='agent_cn', subsets=['cibench_template_cn', 'plugin_eval-mus-p10_one_review_zh']),
    dict(name='agent_en', subsets=['cibench_template', 'plugin_eval-mus-p10_one_review']),
    dict(name='agent', subsets=['agent_cn', 'agent_en']),
]

other_summary_groups = [
    {
        'name': 'average_cn',
        'subsets': [
            ['language_zh_perf_4_and_non_mcq', 'naive_average'],
            ['knowledge_cn', 'perf_4'],
            ['reasonbench_cn_circular', 'perf_circular'],
            ['math_perf_4_and_fill_in_blank_cn', 'naive_average'],
            ['code_cn', 'naive_average'],
            ['agent_cn', 'naive_average'],
        ],
    },
    {
        'name': 'average_en',
        'subsets': [
            ['language_en_perf_4_and_non_mcq', 'naive_average'],
            ['compassbench_v1_knowledge-mixed-cloze_en', 'score'],
            ['reasonbench_en_circular', 'perf_circular'],
            ['math_perf_4_and_fill_in_blank_en', 'naive_average'],
            ['code_en', 'naive_average'],
            ['agent_en', 'naive_average'],
        ],
    },
    {
        'name': 'average',
        'subsets': [
            ['language_perf_4_and_non_mcq', 'naive_average'],
            ['knowledge_perf_4_and_cloze', 'naive_average'],
            ['reasonbench', 'perf_circular'],
            ['math_perf_4_and_fill_in_blank', 'naive_average'],
            ['code', 'naive_average'],
            ['agent', 'naive_average'],
        ],
    },
]




summarizer = dict(
    dataset_abbrs=[
        ['average', 'naive_average'],
        ['average_cn', 'naive_average'],
        ['average_en', 'naive_average'],
        '',
        '',
        '',

        ['language_perf_4_and_non_mcq', 'naive_average'],
        ['language_zh_perf_4_and_non_mcq', 'naive_average'],
        ['language_en_perf_4_and_non_mcq', 'naive_average'],
        ['intention_recognition_zh_circular', 'perf_circular'],
        ['intention_recognition_en_circular', 'perf_circular'],
        ['sentiment_analysis_zh_circular', 'perf_circular'],
        ['sentiment_analysis_en_circular', 'perf_circular'],
        ['translation', 'score'],
        ['content_critic_zh_circular', 'perf_circular'],
        ['content_critic_en_circular', 'perf_circular'],
        ['content_summarization_zh', 'rouge1'],
        ['content_summarization_en', 'rouge1'],
        ['traditional_cultural_understanding_zh_circular', 'perf_circular'],
        ['chinese_semantic_understanding_zh_circular', 'perf_circular'],

        ['knowledge_perf_4_and_cloze', 'naive_average'],
        ['knowledge_cn', 'perf_4'],
        ['compassbench_v1_knowledge-mixed-cloze_en', 'score'],
        ['compassbench_v1_knowledge-common_knowledge-single_choice_cn_circular', 'perf_4'],
        ['compassbench_v1_knowledge-humanity-single_choice_cn_circular', 'perf_4'],
        ['compassbench_v1_knowledge-natural_science-single_choice_cn_circular', 'perf_4'],
        ['compassbench_v1_knowledge-social_science-single_choice_cn_circular', 'perf_4'],

        ['reasonbench', 'perf_circular'],
        ['reasonbench_cn_circular', 'perf_circular'],
        ['reasonbench_en_circular', 'perf_circular'],
        ['reasonbench_cn_commonsense_circular', 'perf_circular'],
        ['reasonbench_cn_abductive_circular', 'perf_circular'],
        ['reasonbench_cn_deductive_circular', 'perf_circular'],
        ['reasonbench_cn_inductive_circular', 'perf_circular'],
        ['reasonbench_en_commonsense_circular', 'perf_circular'],
        ['reasonbench_en_abductive_circular', 'perf_circular'],
        ['reasonbench_en_deductive_logiqa_zh_translated_circular', 'perf_circular'],
        ['reasonbench_en_inductive_circular', 'perf_circular'],

        ['math_perf_4_and_fill_in_blank', 'naive_average'],
        ['math_perf_4_and_fill_in_blank_cn', 'naive_average'],
        ['math_perf_4_and_fill_in_blank_en', 'naive_average'],
        ['compassbench_v1_math-high-single_choice_cn', 'perf_4'],
        ['compassbench_v1_math-high-single_choice_en', 'perf_4'],
        ['compassbench_v1_math-middle-single_choice_cn', 'perf_4'],
        ['compassbench_v1_math-middle-single_choice_en', 'perf_4'],
        ['compassbench_v1_math-primary-cloze_cn', 'accuracy'],
        ['compassbench_v1_math-primary-cloze_en', 'accuracy'],

        ['code', 'naive_average'],
        ['code_cn', 'naive_average'],
        ['code_en', 'naive_average'],
        ['humaneval_cn_pass@1(greedy)', 'naive_average'],
        ['humaneval_plus_pass@1(greedy)', 'naive_average'],
        ['mbpp_cn_pass@1(greedy)', 'naive_average'],
        ['sanitized_mbpp_pass@1(greedy)', 'naive_average'],
        ['humanevalx', 'naive_average'],

        ['agent', 'naive_average'],
        ['agent_cn', 'naive_average'],
        ['agent_en', 'naive_average'],
        ['cibench_template_cn', 'naive_average'],
        ['cibench_template', 'naive_average'],
        ['plugin_eval-mus-p10_one_review_zh', 'naive_average'],
        ['plugin_eval-mus-p10_one_review', 'naive_average'],
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
