from mmengine.config import read_base

with read_base():
    from .groups.cibench import cibench_summary_groups
    from .groups.plugineval import plugineval_summary_groups

agent_summary_groups = [
    # dict(name='math_acc_1_and_fill_in_blank-native', subsets=[['compassbench_v1_math-high-single_choice_cn-native', 'acc_1'], ['compassbench_v1_math-high-single_choice_en-native', 'acc_1'], ['compassbench_v1_math-middle-single_choice_cn-native', 'acc_1'], ['compassbench_v1_math-middle-single_choice_en-native', 'acc_1'], ['compassbench_v1_math-primary-cloze_cn-native', 'accuracy'], ['compassbench_v1_math-primary-cloze_en-native', 'accuracy']]),
    # dict(name='math_perf_4_and_fill_in_blank-native', subsets=[['compassbench_v1_math-high-single_choice_cn-native', 'perf_4'], ['compassbench_v1_math-high-single_choice_en-native', 'perf_4'], ['compassbench_v1_math-middle-single_choice_cn-native', 'perf_4'], ['compassbench_v1_math-middle-single_choice_en-native', 'perf_4'], ['compassbench_v1_math-primary-cloze_cn-native', 'accuracy'], ['compassbench_v1_math-primary-cloze_en-native', 'accuracy']]),
    # dict(name='math_acc_1_and_fill_in_blank-agent', subsets=[['compassbench_v1_math-high-single_choice_cn-agent', 'acc_1'], ['compassbench_v1_math-high-single_choice_en-agent', 'acc_1'], ['compassbench_v1_math-middle-single_choice_cn-agent', 'acc_1'], ['compassbench_v1_math-middle-single_choice_en-agent', 'acc_1'], ['compassbench_v1_math-primary-cloze_cn-agent', 'accuracy'], ['compassbench_v1_math-primary-cloze_en-agent', 'accuracy']]),
    # dict(name='math_perf_4_and_fill_in_blank-agent', subsets=[['compassbench_v1_math-high-single_choice_cn-agent', 'perf_4'], ['compassbench_v1_math-high-single_choice_en-agent', 'perf_4'], ['compassbench_v1_math-middle-single_choice_cn-agent', 'perf_4'], ['compassbench_v1_math-middle-single_choice_en-agent', 'perf_4'], ['compassbench_v1_math-primary-cloze_cn-agent', 'accuracy'], ['compassbench_v1_math-primary-cloze_en-agent', 'accuracy']]),
    # dict(name='agent', subsets=['math_perf_4_and_fill_in_blank-agent', 'cibench_template_wo_nltk:executable', 'cibench_template_wo_nltk:numeric_correct', 'cibench_template_wo_nltk:vis_sim', 'cibench_template_cn_wo_nltk:executable', 'cibench_template_cn_wo_nltk:numeric_correct', 'cibench_template_cn_wo_nltk:vis_sim', 'plugin_eval-p10', 'plugin_eval-p10_zh'], weights={'math_perf_4_and_fill_in_blank-agent': 1, 'cibench_template_wo_nltk:executable': 0.5, 'cibench_template_wo_nltk:numeric_correct': 0.25, 'cibench_template_wo_nltk:vis_sim': 0.25, 'cibench_template_cn_wo_nltk:executable': 0.5, 'cibench_template_cn_wo_nltk:numeric_correct': 0.25, 'cibench_template_cn_wo_nltk:vis_sim': 0.25, 'plugin_eval-p10': 1, 'plugin_eval-p10_zh': 1}),
    dict(name='cibench_template', subsets=['cibench_template:executable', 'cibench_template:numeric_correct', 'cibench_template:text_score', 'cibench_template:vis_sim']),
    dict(name='cibench_template_cn', subsets=['cibench_template_cn:executable', 'cibench_template_cn:numeric_correct', 'cibench_template_cn:text_score', 'cibench_template_cn:vis_sim']),
    dict(name='agent_cn', subsets=['cibench_template_cn', 'plugin_eval-mus-p10_one_review_zh']),
    dict(name='agent_en', subsets=['cibench_template', 'plugin_eval-mus-p10_one_review']),
    dict(name='agent', subsets=['agent_cn', 'agent_en']),
]

summarizer = dict(
    dataset_abbrs=[
        'agent',
        'agent_cn',
        'agent_en',
        'cibench_template_cn',
        'cibench_template',
        'plugin_eval-mus-p10_one_review_zh',
        'plugin_eval-mus-p10_one_review',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], [])
)
