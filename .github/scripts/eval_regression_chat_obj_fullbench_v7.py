from mmengine.config import read_base

with read_base():
    # Datasets
    # Instruct Following
    # # # # Math Calculation
    from opencompass.configs.datasets.aime2024.aime2024_cascade_eval_gen_5e9f4f import \
        aime2024_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f import \
        aime2025_datasets  # noqa: F401, E501
    # # # General Reasoning
    from opencompass.configs.datasets.bbeh.bbeh_llmjudge_gen_86c3a0 import \
        bbeh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_complete_gen_2888d3 import \
        bigcodebench_hard_complete_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_instruct_gen_c3d5ad import \
        bigcodebench_hard_instruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.chem_exam.competition_gen import \
        chem_competition_instruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.chem_exam.gaokao_gen import \
        chem_gaokao_instruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ChemBench.ChemBench_llmjudge_gen_c584cf import \
        chembench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ClimaQA.ClimaQA_Gold_llm_judge_gen_f15343 import \
        climaqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmmlu.cmmlu_llmjudge_gen_e1cd9a import \
        cmmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.Earth_Silver.Earth_Silver_llmjudge_gen import \
        earth_silver_mcq_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_gen_772ea0 import \
        gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.HLE.hle_llmverify_gen_6ff468 import \
        hle_datasets  # noqa: F401, E501
    # # Coding
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_dcae0e import \
        humaneval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import \
        ifeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.kcle.kcle_llm_judge_gen import \
        kcle_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.korbench.korbench_single_0shot_cascade_eval_gen_56cf43 import \
        korbench_0shot_single_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import \
        LCBCodeGeneration_dataset  # noqa: F401, E501
    from opencompass.configs.datasets.livemathbench.livemathbench_hard_custom_cascade_eval_gen_4bce59 import \
        livemathbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.matbench.matbench_llm_judge_gen_0e9276 import \
        matbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_500_cascade_eval_gen_6ff468 import \
        math_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import \
        sanitized_mbpp_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MedXpertQA.MedXpertQA_llmjudge_gen import \
        medxpertqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_llmjudge_gen_f4336b import \
        mmlu_datasets  # noqa: F401, E501
    # # # Knowledge
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_nocot_genericllmeval_gen_08c1de import \
        mmlu_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.OlymMATH.olymmath_llmverify_gen_97b203 import \
        olymmath_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.OlympiadBench.OlympiadBench_0shot_llmverify_gen_be8b13 import \
        olympiadbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.PHYBench.phybench_gen import \
        phybench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.PHYSICS.PHYSICS_llm_judge_gen_a133a2 import \
        physics_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ProteinLMBench.ProteinLMBench_llmjudge_gen_a67965 import \
        proteinlmbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.R_Bench.rbench_llmjudge_gen_c89350 import \
        RBench_datasets  # noqa: F401, E501
    #     # Academic
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_0shot_instruct_gen import \
        smolinstruct_datasets_0shot_instruct as \
        smolinstruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.srbench.srbench_gen import \
        srbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.supergpqa.supergpqa_cascade_gen_1545c1 import \
        supergpqa_datasets  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm3_8b_instruct import \
        models as hf_internlm3_8b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm3_8b_instruct import \
        models as lmdeploy_internlm3_8b_instruct_model  # noqa: F401, E501
    # Summary Groups
    from opencompass.configs.summarizers.groups.bbeh import \
        bbeh_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.cmmlu import \
        cmmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.korbench import \
        korbench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu import \
        mmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu_pro import \
        mmlu_pro_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.OlympiadBench import \
        OlympiadBenchPhysics_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.OlympiadBench import (  # noqa: F401, E501
        OlympiadBench_summary_groups, OlympiadBenchMath_summary_groups)
    from opencompass.configs.summarizers.groups.PHYSICS import \
        physics_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.supergpqa import \
        supergpqa_summary_groups  # noqa: F401, E501

    from ...volc import infer  # noqa: F401, E501

# Add lattest LCB version
LCBCodeGeneration_v6_datasets = LCBCodeGeneration_dataset
LCBCodeGeneration_v6_datasets['abbr'] = 'lcb_code_generation_v6'
LCBCodeGeneration_v6_datasets['release_version'] = 'v6'
LCBCodeGeneration_v6_datasets['eval_cfg']['evaluator'][
    'release_version'] = 'v6'
LCBCodeGeneration_v6_datasets = [LCBCodeGeneration_v6_datasets]

repeated_info = [
    (math_datasets, 1),
    (gpqa_datasets, 1),
    (aime2024_datasets, 1),
    (aime2025_datasets, 1),
    (olympiadbench_datasets, 1),
    (livemathbench_datasets, 1),
    (olymmath_datasets, 1),
    (korbench_0shot_single_datasets, 1),
]

for datasets_, num in repeated_info:
    for dataset_ in datasets_:
        dataset_['n'] = num
        dataset_['k'] = num

datasets = [
    v[0] for k, v in locals().items() if k.endswith('_datasets')
    and 'bigcode' not in k.lower() and isinstance(v, list) and len(v) > 0
]

datasets += bigcodebench_hard_instruct_datasets
datasets += bigcodebench_hard_complete_datasets

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
