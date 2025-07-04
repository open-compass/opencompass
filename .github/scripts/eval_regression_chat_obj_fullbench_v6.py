from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.aime2024.aime2024_llmjudge_gen_5e9f4f import \
        aime2024_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.aime2025.aime2025_llmjudge_gen_5e9f4f import \
        aime2025_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ARC_Prize_Public_Evaluation.arc_prize_public_evaluation_gen_fedd04 import \
        arc_prize_public_evaluation_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bbh.bbh_llmjudge_gen_b5bdf1 import \
        bbh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmo_fib.cmo_fib_gen_2783e5 import \
        cmo_fib_datasets  # noqa: F401, E501
    # dingo
    from opencompass.configs.datasets.dingo.dingo_gen import \
        datasets as dingo_datasets  # noqa: F401, E501
    # General Reasoning
    from opencompass.configs.datasets.drop.drop_llmjudge_gen_3857b0 import \
        drop_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_d16acb import \
        GaokaoBench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_0shot_nocot_genericllmeval_gen_772ea0 import \
        gpqa_datasets  # noqa: F401, E501
    # Math Calculation
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hellaswag.hellaswag_llmjudge_gen_809ef1 import \
        hellaswag_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.korbench.korbench_llmjudge_gen_56cf43 import \
        korbench_0shot_single_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_500_llmjudge_gen_6ff468 import \
        math_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MathBench.mathbench_2024_gen_4b8f28 import \
        mathbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.musr.musr_llmjudge_gen_b47fd3 import \
        musr_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.supergpqa.supergpqa_llmjudge_gen_12b8bc import \
        supergpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_c87d61 import \
        triviaqa_datasets  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm3_8b_instruct import \
        models as hf_internlm3_8b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm3_8b_instruct import \
        models as lmdeploy_internlm3_8b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_32b_instruct import \
        models as lmdeploy_qwen2_5_32b_instruct  # noqa: F401, E501
    # Summary Groups
    from opencompass.configs.summarizers.groups.bbeh import \
        bbeh_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.bbh import \
        bbh_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.cmmlu import \
        cmmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.GaokaoBench import \
        GaokaoBench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.korbench import \
        korbench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mathbench_v1_2024 import \
        mathbench_2024_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu import \
        mmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu_pro import \
        mmlu_pro_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.musr_average import \
        summarizer as musr_summarizer
    from opencompass.configs.summarizers.mmmlu_lite import \
        mmmlu_summary_groups  # noqa: F401, E501

    from ...volc import infer  # noqa: F401, E501

datasets = [
    v[0] for k, v in locals().items() if k.endswith('_datasets')
    and 'scicode' not in k.lower() and 'dingo' not in k.lower()
    and 'arc_prize' not in k.lower() and isinstance(v, list) and len(v) > 0
]

dingo_datasets[0]['abbr'] = 'qa_dingo_cn'
dingo_datasets[0]['path'] = 'data/qabench/history_prompt_case_cn.csv'
datasets.append(dingo_datasets[0])
datasets += arc_prize_public_evaluation_datasets

musr_summary_groups = musr_summarizer['summary_groups']
summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], [])

summary_groups.append(
    {
        'name': 'Mathbench',
        'subsets': ['mathbench-a (average)', 'mathbench-t (average)'],
    }, )

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:16]'
    if 'dataset_cfg' in d['eval_cfg']['evaluator'] and 'reader_cfg' in d[
            'eval_cfg']['evaluator']['dataset_cfg']:
        d['eval_cfg']['evaluator']['dataset_cfg']['reader_cfg'][
            'test_range'] = '[0:16]'
    if 'llm_evaluator' in d['eval_cfg']['evaluator'] and 'dataset_cfg' in d[
            'eval_cfg']['evaluator']['llm_evaluator']:
        d['eval_cfg']['evaluator']['llm_evaluator']['dataset_cfg'][
            'reader_cfg']['test_range'] = '[0:16]'

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
for m in models:
    m['abbr'] = m['abbr'] + '_fullbench'
    if 'turbomind' in m['abbr'] or 'lmdeploy' in m['abbr']:
        m['engine_config']['max_batch_size'] = 1
        m['batch_size'] = 1

models = sorted(models, key=lambda x: x['run_cfg']['num_gpus'])

obj_judge_model = lmdeploy_internlm3_8b_instruct_model[0]
obj_judge_model['engine_config']['max_batch_size'] = 1
obj_judge_model['engine_config']['cache_max_entry_count'] = 0.6
obj_judge_model['batch_size'] = 1

for d in datasets:
    if 'judge_cfg' in d['eval_cfg']['evaluator']:
        d['eval_cfg']['evaluator']['judge_cfg'] = obj_judge_model
    if 'llm_evaluator' in d['eval_cfg']['evaluator'] and 'judge_cfg' in d[
            'eval_cfg']['evaluator']['llm_evaluator']:
        d['eval_cfg']['evaluator']['llm_evaluator'][
            'judge_cfg'] = obj_judge_model
