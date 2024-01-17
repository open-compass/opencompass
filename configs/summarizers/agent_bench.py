from mmengine.config import read_base

with read_base():
    from .groups.cibench import cibench_summary_groups
    from .groups.plugineval import plugineval_summary_groups

agent_summary_groups = [
    dict(name='math_acc_1_and_fill_in_blank-native', subsets=[['compassbench_v1_math-high-single_choice_cn-native', 'acc_1'], ['compassbench_v1_math-high-single_choice_en-native', 'acc_1'], ['compassbench_v1_math-middle-single_choice_cn-native', 'acc_1'], ['compassbench_v1_math-middle-single_choice_en-native', 'acc_1'], ['compassbench_v1_math-primary-cloze_cn-native', 'accuracy'], ['compassbench_v1_math-primary-cloze_en-native', 'accuracy']]),
    dict(name='math_perf_4_and_fill_in_blank-native', subsets=[['compassbench_v1_math-high-single_choice_cn-native', 'perf_4'], ['compassbench_v1_math-high-single_choice_en-native', 'perf_4'], ['compassbench_v1_math-middle-single_choice_cn-native', 'perf_4'], ['compassbench_v1_math-middle-single_choice_en-native', 'perf_4'], ['compassbench_v1_math-primary-cloze_cn-native', 'accuracy'], ['compassbench_v1_math-primary-cloze_en-native', 'accuracy']]),
    dict(name='math_acc_1_and_fill_in_blank-agent', subsets=[['compassbench_v1_math-high-single_choice_cn-agent', 'acc_1'], ['compassbench_v1_math-high-single_choice_en-agent', 'acc_1'], ['compassbench_v1_math-middle-single_choice_cn-agent', 'acc_1'], ['compassbench_v1_math-middle-single_choice_en-agent', 'acc_1'], ['compassbench_v1_math-primary-cloze_cn-agent', 'accuracy'], ['compassbench_v1_math-primary-cloze_en-agent', 'accuracy']]),
    dict(name='math_perf_4_and_fill_in_blank-agent', subsets=[['compassbench_v1_math-high-single_choice_cn-agent', 'perf_4'], ['compassbench_v1_math-high-single_choice_en-agent', 'perf_4'], ['compassbench_v1_math-middle-single_choice_cn-agent', 'perf_4'], ['compassbench_v1_math-middle-single_choice_en-agent', 'perf_4'], ['compassbench_v1_math-primary-cloze_cn-agent', 'accuracy'], ['compassbench_v1_math-primary-cloze_en-agent', 'accuracy']]),
    dict(
        name='agent',
        subsets=['math_perf_4_and_fill_in_blank-agent', 'cibench_template_wo_nltk:executable', 'cibench_template_wo_nltk:numeric_correct', 'cibench_template_wo_nltk:vis_sim', 'cibench_template_cn_wo_nltk:executable', 'cibench_template_cn_wo_nltk:numeric_correct', 'cibench_template_cn_wo_nltk:vis_sim', 'plugin_eval-p10', 'plugin_eval-p10_zh'],
        weights={'math_perf_4_and_fill_in_blank-agent': 1, 'cibench_template_wo_nltk:executable': 0.5, 'cibench_template_wo_nltk:numeric_correct': 0.25, 'cibench_template_wo_nltk:vis_sim': 0.25, 'cibench_template_cn_wo_nltk:executable': 0.5, 'cibench_template_cn_wo_nltk:numeric_correct': 0.25, 'cibench_template_cn_wo_nltk:vis_sim': 0.25, 'plugin_eval-p10': 1, 'plugin_eval-p10_zh': 1}
    )
]

summarizer = dict(
    dataset_abbrs=[
        'agent',
        'math_acc_1_and_fill_in_blank-native',
        'math_perf_4_and_fill_in_blank-native',
        # '######## MathBench-Agent Accuracy ########', # category
        'math_acc_1_and_fill_in_blank-agent',
        'math_perf_4_and_fill_in_blank-agent',
        # '######## CIBench Template ########', # category
        'cibench_template:executable',
        'cibench_template:numeric_correct',
        'cibench_template:text_score',
        'cibench_template:vis_sim',
        # '######## CIBench Template Chinese ########', # category
        'cibench_template_cn:executable',
        'cibench_template_cn:numeric_correct',
        'cibench_template_cn:text_score',
        'cibench_template_cn:vis_sim',
        # '######## CIBench Template w/o NLTK ########', # category no text score becase it is only for nltk
        'cibench_template_wo_nltk:executable',
        'cibench_template_wo_nltk:numeric_correct',
        'cibench_template_wo_nltk:vis_sim',
        # '######## CIBench Template Chinese w/o NLTK ########', # category
        'cibench_template_cn_wo_nltk:executable',
        'cibench_template_cn_wo_nltk:numeric_correct',
        'cibench_template_cn_wo_nltk:vis_sim',
        # '######## T-Eval ########', # category
        ['plugin_eval-p10', 'naive_average'],
        ['plugin_eval-p10-instruct_v1', 'format_metric'],
        ['plugin_eval-p10-instruct_v1', 'args_em_metric'],
        ['plugin_eval-p10-plan_str_v1', 'f1_score'],
        ['plugin_eval-p10-plan_json_v1', 'f1_score'],
        ['plugin_eval-p10-reason_str_v1', 'thought'],
        ['plugin_eval-p10-reason_retrieve_understand_json_v1', 'thought'],
        ['plugin_eval-p10-retrieve_str_v1', 'name'],
        ['plugin_eval-p10-reason_retrieve_understand_json_v1', 'name'],
        ['plugin_eval-p10-understand_str_v1', 'args'],
        ['plugin_eval-p10-reason_retrieve_understand_json_v1', 'args'],
        ['plugin_eval-p10-review_str_v1', 'review_quality'],

        ['plugin_eval-p10_zh', 'naive_average'],
        ['plugin_eval-p10-instruct_v1_zh', 'format_metric'],
        ['plugin_eval-p10-instruct_v1_zh', 'args_em_metric'],
        ['plugin_eval-p10-plan_str_v1_zh', 'f1_score'],
        ['plugin_eval-p10-plan_json_v1_zh', 'f1_score'],
        ['plugin_eval-p10-reason_str_v1_zh', 'thought'],
        ['plugin_eval-p10-reason_retrieve_understand_json_v1_zh', 'thought'],
        ['plugin_eval-p10-retrieve_str_v1_zh', 'name'],
        ['plugin_eval-p10-reason_retrieve_understand_json_v1_zh', 'name'],
        ['plugin_eval-p10-understand_str_v1_zh', 'args'],
        ['plugin_eval-p10-reason_retrieve_understand_json_v1_zh', 'args'],
        ['plugin_eval-p10-review_str_v1_zh', 'review_quality'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], [])
)
