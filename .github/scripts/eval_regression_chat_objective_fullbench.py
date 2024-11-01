from mmengine.config import read_base

with read_base():
    # read hf models - chat models
    # Dataset
    from opencompass.configs.datasets.ARC_c.ARC_c_few_shot_ppl import \
        ARC_c_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bbh.bbh_gen_98fba6 import \
        bbh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmmlu.cmmlu_ppl_041cbf import \
        cmmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.drop.drop_gen_a2697c import \
        drop_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_d21e37 import \
        GaokaoBench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_few_shot_ppl_4b5a83 import \
        gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_ppl_59c85e import \
        hellaswag_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.humaneval.internal_humaneval_gen_ce6b06 import \
        humaneval_datasets as humaneval_v2_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.humaneval.internal_humaneval_gen_d2537e import \
        humaneval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_4shot_base_gen_43d5b6 import \
        math_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MathBench.mathbench_2024_few_shot_mixed_4a3fd4 import \
        mathbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_742f0c import \
        sanitized_mbpp_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import \
        mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_few_shot_gen_bfaf90 import \
        mmlu_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.nq.nq_open_1shot_gen_20a989 import \
        nq_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_few_shot_ppl import \
        race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_few_shot_ppl import \
        BoolQ_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import \
        TheoremQA_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_20a989 import \
        triviaqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.wikibench.wikibench_few_shot_ppl_c23d79 import \
        wikibench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.winogrande.winogrande_5shot_ll_252f01 import \
        winogrande_datasets  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_7b_chat import \
        models as hf_internlm2_5_7b_chat_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b_chat import \
        models as lmdeploy_internlm2_5_7b_chat_model  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.bbh import \
        bbh_summary_groups  # noqa: F401, E501
    # Summary Groups
    from opencompass.configs.summarizers.groups.cmmlu import \
        cmmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.GaokaoBench import \
        GaokaoBench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mathbench_v1_2024 import \
        mathbench_2024_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu import \
        mmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu_pro import \
        mmlu_pro_summary_groups  # noqa: F401, E501

race_datasets = [race_datasets[1]]
humaneval_v2_datasets[0]['abbr'] = 'openai_humaneval_v2'
datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

summarizer = dict(
    dataset_abbrs=[
        ['race-high', 'accuracy'],
        ['ARC-c', 'accuracy'],
        ['BoolQ', 'accuracy'],
        ['mmlu_pro', 'naive_average'],
        ['GPQA_diamond', 'accuracy'],
        ['cmmlu', 'naive_average'],
        ['mmlu', 'naive_average'],
        ['drop', 'accuracy'],
        ['bbh', 'naive_average'],
        ['math', 'accuracy'],
        ['openai_humaneval', 'humaneval_pass@1'],
        ['openai_humaneval_v2', 'humaneval_pass@1'],
        ['sanitized_mbpp', 'score'],
        ['wikibench-wiki-single_choice_cncircular', 'perf_4'],
        ['gsm8k', 'accuracy'],
        ['GaokaoBench', 'weighted_average'],
        ['triviaqa_wiki_1shot', 'score'],
        ['nq_open_1shot', 'score'],
        ['winogrande', 'accuracy'],
        ['hellaswag', 'accuracy'],
        ['TheoremQA', 'score'],
        '###### MathBench-A: Application Part ######',
        'college',
        'high',
        'middle',
        'primary',
        'arithmetic',
        'mathbench-a (average)',
        '###### MathBench-T: Theory Part ######',
        'college_knowledge',
        'high_knowledge',
        'middle_knowledge',
        'primary_knowledge',
        'mathbench-t (average)',
        '###### Overall: Average between MathBench-A and MathBench-T ######',
        'Overall',
        '',
        'mmlu',
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-humanities',
        'mmlu-other',
        'cmmlu',
        'cmmlu-stem',
        'cmmlu-social-science',
        'cmmlu-humanities',
        'cmmlu-other',
        'cmmlu-china-specific',
        'mmlu_pro',
        'mmlu_pro_biology',
        'mmlu_pro_business',
        'mmlu_pro_chemistry',
        'mmlu_pro_computer_science',
        'mmlu_pro_economics',
        'mmlu_pro_engineering',
        'mmlu_pro_health',
        'mmlu_pro_history',
        'mmlu_pro_law',
        'mmlu_pro_math',
        'mmlu_pro_philosophy',
        'mmlu_pro_physics',
        'mmlu_pro_psychology',
        'mmlu_pro_other',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

for model in [
        v for k, v in locals().items()
        if k.endswith('_model') and 'lmdeploy' in k
]:
    model['engine_config']['max_batch_size'] = 1
    model['batch_size'] = 1

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:10]'

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
