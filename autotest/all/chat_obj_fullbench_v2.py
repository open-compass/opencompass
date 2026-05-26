"""Infer config: chat_objective rawprompt + chatml datasets."""
from mmengine.config import read_base

with read_base():
    # Infer with dict(round=...) meta_template; list meta (raw_template_models)
    # breaks GenInferencer + API parse_template on rawprompt datasets.
    from autotest.all.config import common_infer as infer  # noqa: F401, E501
    from autotest.all.config import \
        raw_template_models as models  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.AMO_Bench.AMO_Bench_gen import \
        datasets as AMO_Bench_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.C_MHChem.C_MHChem_gen import \
        datasets as C_MHChem_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.CPsyExam.CPsyExam_gen import \
        datasets as CPsyExam_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.CS_Bench.CS_Bench_gen import \
        datasets as CS_Bench_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.HMMT2025.HMMT2025_repeat32_gen import \
        datasets as HMMT2025_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.IMO_Bench_AnswerBench.IMO_Bench_AnswerBench_gen import \
        datasets as IMO_Bench_AnswerBench_chatml  # noqa: F401, E501
    # ChatML Datasets
    from opencompass.configs.chatml_datasets.MaScQA.MaScQA_gen import \
        datasets as MaScQA_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.UGD_hard.UGD_hard_repeat8_gen import \
        datasets as UGD_hard_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.UGPhysics.UGPhysics_gen import \
        datasets as UGPhysics_chatml  # noqa: F401, E501
    # Math Calculation
    from opencompass.configs.datasets.aime2024.aime2024_cascade_eval_rawprompt_gen_2f2c96 import \
        aime2024_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_rawprompt_gen_2f2c96 import \
        aime2025_datasets  # noqa: F401, E501
    # CompassAcademic Extended
    from opencompass.configs.datasets.aime2025.aime2025_llmjudge_academic_rawprompt import \
        aime2025_datasets as \
        CompassAcademic_aime2025_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.aime2026.aime2026_cascade_eval_rawprompt_gen_0970dd import \
        aime2026_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.atlas.atlas_val_rawprompt_gen_277bee import \
        atlas_datasets  # noqa: F401, E501
    # General Reasoning
    from opencompass.configs.datasets.bbeh.bbeh_llmjudge_rawprompt_gen_36b5f4 import \
        bbeh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_complete_rawprompt_gen_95140b import \
        bigcodebench_hard_complete_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_instruct_rawprompt_gen_5cbb9f import \
        bigcodebench_hard_instruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.biodata.biodata_task_rawprompt_gen import \
        biodata_task_datasets  # noqa: F401, E501
    # from opencompass.configs.datasets.CARDBiomedBench.CARDBiomedBench_llmjudge_rawprompt_gen_b4d90c import \ # noqa: F401, E501
    #    cardbiomedbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.chem_exam.competition_rawprompt_gen import \
        chem_competition_instruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.chem_exam.gaokao_rawprompt_gen import \
        chem_gaokao_instruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ChemBench.ChemBench_llmjudge_rawprompt_gen_fa3fc4 import \
        chembench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ClimaQA.ClimaQA_Gold_llm_judge_rawprompt_gen_b3080f import \
        climaqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmmlu.cmmlu_llmjudge_rawprompt_gen_9f9c31 import \
        cmmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.CMPhysBench.cmphysbench_rawprompt_gen import \
        cmphysbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.Earth_Silver.Earth_Silver_llmjudge_rawprompt_gen_a84bc6 import \
        earth_silver_mcq_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_academic_rawprompt import \
        gpqa_datasets as CompassAcademic_gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_rawprompt_gen_706039 import \
        gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.HLE.hle_llmverify_academic_rawprompt import \
        hle_datasets as CompassAcademic_hle_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.HLE.hle_llmverify_rawprompt_gen_0970dd import \
        hle_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hmmt2026.hmmt2026_cascade_eval_rawprompt_gen_0970dd import \
        hmmt2026_datasets  # noqa: F401, E501
    # Coding
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_rawprompt_gen_6ce2ca import \
        humaneval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFBench.IFBench_rawprompt_gen import \
        ifbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFEval.IFEval_rawprompt_gen_e7f781 import \
        ifeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.kcle.kcle_llm_judge_rawprompt_gen_16e383 import \
        kcle_datasets as kcle_fix_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.korbench.korbench_single_0shot_cascade_eval_rawprompt_gen_c048da import \
        korbench_0shot_single_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.livecodebench.livecodebench_rawprompt_gen_c09673 import \
        LCBCodeGeneration_dataset  # noqa: F401, E501
    from opencompass.configs.datasets.livecodebench.livecodebench_v6_academic_rawprompt import \
        LCBCodeGeneration_dataset as \
        CompassAcademic_LCBCodeGeneration_dataset  # noqa: F401, E501
    from opencompass.configs.datasets.livecodebench_pro.livecodebench_pro_rawprompt_gen import \
        lcb_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.livemathbench.livemathbench_hard_custom_cascade_eval_rawprompt_gen_e1ce64 import \
        livemathbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.matbench.matbench_llm_judge_rawprompt_gen_c987b6 import \
        matbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_500_cascade_eval_rawprompt_gen_0970dd import \
        math_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_0shot_nocot_rawprompt_gen_30c1e5 import \
        sanitized_mbpp_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.medmcqa.medmcqa_llmjudge_rawprompt_gen_015178 import \
        medmcqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MedXpertQA.MedXpertQA_llmjudge_rawprompt_gen import \
        medxpertqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_llmjudge_rawprompt_gen_af67f0 import \
        mmlu_datasets  # noqa: F401, E501
    # Knowledge
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_nocot_genericllmeval_rawprompt_gen_0321fb import \
        mmlu_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.molculariq.molculariq_rawprompt_gen import \
        moleculariq_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MolInstructions_chem.mol_instructions_chem_rawprompt_gen import \
        mol_gen_selfies_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.OlymMATH.olymmath_llmverify_rawprompt_gen_9d3a8e import \
        olymmath_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.OlympiadBench.OlympiadBench_0shot_llmverify_rawprompt_gen_d3e9e4 import \
        olympiadbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.openswi.openswi_rawprompt_gen import \
        openswi_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.PHYBench.phybench_rawprompt_gen import \
        phybench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.PHYSICS.PHYSICS_llm_judge_rawprompt_gen_56ebc8 import \
        physics_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ProteinLMBench.ProteinLMBench_llmjudge_rawprompt_gen_9627a6 import \
        proteinlmbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.R_Bench.rbench_llmjudge_rawprompt_gen_c24221 import \
        RBench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SimpleQA.simpleqa_verified_rawprompt_gen import \
        simpleqa_verified_datasets  # noqa: F401, E501
    # AI4S
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_0shot_instruct_rawprompt_gen import \
        mini_smolinstruct_datasets_0shot_instruct as \
        mini_smolinstruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_0shot_instruct_rawprompt_gen import \
        smolinstruct_datasets_0shot_instruct as \
        smolinstruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.srbench.srbench_rawprompt_gen import \
        srbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.supergpqa.supergpqa_cascade_rawprompt_gen_ca8345 import \
        supergpqa_datasets  # noqa: F401, E501
    # Summary Groups
    from opencompass.configs.summarizers.groups.bbeh import \
        bbeh_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.biodata import \
        biodata_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.cmmlu import \
        cmmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.korbench import \
        korbench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.matbench import \
        matbench_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu import \
        mmlu_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mmlu_pro import \
        mmlu_pro_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.mol_instruct import \
        mol_instruct_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.OlympiadBench import \
        OlympiadBenchPhysics_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.OlympiadBench import (  # noqa: F401, E501
        OlympiadBench_summary_groups, OlympiadBenchMath_summary_groups)
    from opencompass.configs.summarizers.groups.PHYSICS import \
        physics_summary_groups  # noqa: F401, E501
    from opencompass.configs.summarizers.groups.smolinstruct import (  # noqa: F401, E501
        smolinstruct_mini_summary_groups, smolinstruct_summary_groups)
    from opencompass.configs.summarizers.groups.supergpqa import \
        supergpqa_summary_groups  # noqa: F401, E501

LCBCodeGeneration_v6_datasets = __import__('copy').deepcopy(
    LCBCodeGeneration_dataset)
LCBCodeGeneration_v6_datasets['abbr'] = 'lcb_code_generation_v6'
LCBCodeGeneration_v6_datasets['release_version'] = 'v6'
LCBCodeGeneration_v6_datasets['eval_cfg']['evaluator'][
    'release_version'] = 'v6'
LCBCodeGeneration_v6_datasets['eval_cfg']['evaluator'][
    'extractor_version'] = 'v2'
LCBCodeGeneration_v6_datasets = [LCBCodeGeneration_v6_datasets]

LCBCodeGeneration_v5_datasets = __import__('copy').deepcopy(
    LCBCodeGeneration_dataset)
LCBCodeGeneration_v5_datasets['abbr'] = 'lcb_code_generation_v5'
LCBCodeGeneration_v5_datasets['release_version'] = 'v5'
LCBCodeGeneration_v5_datasets['eval_cfg']['evaluator'][
    'release_version'] = 'v5'
LCBCodeGeneration_v5_datasets['eval_cfg']['evaluator'][
    'extractor_version'] = 'v2'
LCBCodeGeneration_v5_datasets = [LCBCodeGeneration_v5_datasets]

CompassAcademic_LCBCodeGeneration_datasets = [
    __import__('copy').deepcopy(CompassAcademic_LCBCodeGeneration_dataset),
]

cmphysbench_datasets[0]['abbr'] = cmphysbench_datasets[0]['abbr'] + '_repeat2'
UGD_hard_chatml[0]['abbr'] = 'UGD_hard_repeat2'
HMMT2025_chatml[0]['n'] = 2
HMMT2025_chatml[0]['abbr'] = 'HMMT2025_repeat2'
math_datasets[0]['abbr'] = 'math500_prm800k'

compassacademic_dataset_list = [
    CompassAcademic_aime2025_datasets,
    CompassAcademic_gpqa_datasets,
    CompassAcademic_LCBCodeGeneration_datasets,
    CompassAcademic_hle_datasets,
]
for acadatasets in compassacademic_dataset_list:
    for acadataset in acadatasets:
        acadataset['abbr'] = acadataset['abbr'] + '_CompassAcademic'
        acadataset['n'] = 2

repeated_info = [
    (math_datasets, 2),
    (gpqa_datasets, 2),
    (aime2024_datasets, 2),
    (aime2025_datasets, 2),
    (olympiadbench_datasets, 1),
    (livemathbench_datasets, 2),
    (olymmath_datasets, 2),
    (korbench_0shot_single_datasets, 2),
    (cmphysbench_datasets, 2),
    (aime2026_datasets, 2),
    (hmmt2026_datasets, 2),
]

for datasets_, num in repeated_info:
    for dataset_ in datasets_:
        dataset_['n'] = num
        dataset_['k'] = num

# Multi-subject *_datasets: mock/CI only keep the first subset.
_FIRST_SUBSET_ONLY_DATASETS = frozenset({
    'mmlu_datasets',
    'mmlu_pro_datasets',
    'cmmlu_datasets',
    'chembench_datasets',
    'bbeh_datasets',
    'matbench_datasets',
    'climaqa_datasets',
    'cardbiomedbench_datasets',
    'supergpqa_datasets',
    'medmcqa_datasets',
    'medxpertqa_datasets',
    'proteinlmbench_datasets',
    'moleculariq_datasets',
    'physics_datasets',
    'RBench_datasets',
    'earth_silver_mcq_datasets',
    'korbench_0shot_single_datasets',
    'UGPhysics_chatml',
})

datasets = sum(
    ([v[0]] if v else [] if k in _FIRST_SUBSET_ONLY_DATASETS else v
     for k, v in locals().items() if k.endswith('_datasets')),
    [],
)
chatml_datasets = sum(
    ([v[0]] if v else [] if k in _FIRST_SUBSET_ONLY_DATASETS else v
     for k, v in locals().items() if k.endswith('_chatml')),
    [],
)

summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')],
    [],
)
summary_groups.extend([
    {
        'name':
        'olymmath_llmjudge',
        'subsets': [
            'olymmath_llmjudge_en-hard',
            'olymmath_llmjudge_zh-hard',
            'olymmath_llmjudge_en-easy',
            'olymmath_llmjudge_zh-easy',
        ],
    },
    {
        'name':
        'ChemBench',
        'subsets': [
            'ChemBench_Property_Prediction',
            'ChemBench_Mol2caption',
            'ChemBench_Caption2mol',
            'ChemBench_Product_Prediction',
            'ChemBench_Retrosynthesis',
            'ChemBench_Yield_Prediction',
            'ChemBench_Temperature_Prediction',
        ],
    },
    {
        'name':
        'ClimaQA',
        'subsets': [
            'ClimaQA_Gold_cloze',
            'ClimaQA_Gold_ffq',
            'ClimaQA_Gold_mcq',
        ],
    },
    {
        'name': 'OlympiadBench',
        'subsets': ['OlympiadBenchMath', 'OlympiadBenchPhysics'],
    },
    {
        'name':
        'livemathbench_hard',
        'subsets': [
            'livemathbench_hard_custom_hard_cn',
            'livemathbench_hard_custom_hard_en',
        ],
    },
    {
        'name': 'R-Bench',
        'subsets': ['R-Bench_en', 'R-Bench_zh'],
    },
    {
        'name': 'Chem_exam',
        'subsets': ['Chem_exam-competition', 'Chem_exam-gaokao'],
    },
    {
        'name':
        'MolecularIQ',
        'subsets': [
            'MolecularIQ-count',
            'MolecularIQ-generation',
            'MolecularIQ-index',
        ],
    },
    {
        'name':
        'UGPhysics',
        'subsets': [
            'UGPhysics_AtomicPhysics_zh',
            'UGPhysics_AtomicPhysics_en',
            'UGPhysics_ClassicalElectromagnetism_zh',
            'UGPhysics_ClassicalElectromagnetism_en',
            'UGPhysics_ClassicalMechanics_zh',
            'UGPhysics_ClassicalMechanics_en',
            'UGPhysics_Electrodynamics_zh',
            'UGPhysics_Electrodynamics_en',
            'UGPhysics_GeometricalOptics_zh',
            'UGPhysics_GeometricalOptics_en',
            'UGPhysics_QuantumMechanics_zh',
            'UGPhysics_QuantumMechanics_en',
            'UGPhysics_Relativity_zh',
            'UGPhysics_Relativity_en',
            'UGPhysics_Solid-StatePhysics_zh',
            'UGPhysics_Solid-StatePhysics_en',
            'UGPhysics_StatisticalMechanics_zh',
            'UGPhysics_StatisticalMechanics_en',
            'UGPhysics_SemiconductorPhysics_zh',
            'UGPhysics_SemiconductorPhysics_en',
            'UGPhysics_Thermodynamics_zh',
            'UGPhysics_Thermodynamics_en',
            'UGPhysics_TheoreticalMechanics_zh',
            'UGPhysics_TheoreticalMechanics_en',
            'UGPhysics_WaveOptics_zh',
            'UGPhysics_WaveOptics_en',
        ],
    },
])

# Summarizer
summarizer = dict(
    dataset_abbrs=[
        'Knowledge',
        ['mmlu_pro', 'accuracy'],
        ['cmmlu', 'accuracy'],
        ['mmlu', 'accuracy'],
        ['simpleqa-verified', 'accuracy'],
        '',
        'Instruction Following',
        ['IFEval', 'Prompt-level-strict-accuracy'],
        ['IFBench', 'score'],
        '',
        'General Reasoning',
        ['bbeh', 'naive_average'],
        ['bbeh', 'harmonic_mean'],
        ['GPQA_diamond', 'accuracy (8 runs average)'],
        ['GPQA_diamond', 'G-Pass@8_0.0'],
        ['korbench_single', 'accuracy (4 runs average)'],
        ['korbench_single', 'G-Pass@4_0.0'],
        ['SuperGPQA', 'weighted_average'],
        ['hle_llmjudge', 'accuracy'],
        ['OlympiadBench', 'accuracy'],
        '',
        'Math Calculation',
        ['aime2024', 'accuracy (32 runs average)'],
        ['aime2024', 'G-Pass@32_0.0'],
        ['aime2025', 'accuracy (32 runs average)'],
        ['aime2025', 'G-Pass@32_0.0'],
        ['aime2026', 'accuracy (32 runs average)'],
        ['aime2026', 'G-Pass@32_0.0'],
        ['hmmt2026', 'accuracy (32 runs average)'],
        ['hmmt2026', 'G-Pass@32_0.0'],
        ['livemathbench_hard', 'accuracy (32 runs average)'],
        ['livemathbench_hard', 'G-Pass@32_0.0'],
        ['olymmath_llmjudge', 'naive_average'],
        ['math500_prm800k', 'accuracy (4 runs average)'],
        ['math500_prm800k', 'G-Pass@4_0.0'],
        '',
        'Code',
        ['openai_humaneval', 'humaneval_pass@1'],
        ['sanitized_mbpp', 'score'],
        ['lcb_code_generation_v5', 'pass@1'],
        ['lcb_code_generation_v6', 'pass@1'],
        ['bigcodebench_hard_instruct', 'pass@1'],
        ['bigcodebench_hard_complete', 'pass@1'],
        ['lcb_pro', 'pass@1'],
        '',
        'AI4S',
        ['medxpertqa', 'accuracy'],
        ['ChemBench', 'naive_average'],
        ['ClimaQA', 'naive_average'],
        ['PHYSICS', 'naive_average'],
        ['phybench-eed', 'accuracy'],
        ['ProteinLMBench', 'accuracy'],
        ['earth_silver_mcq', 'accuracy'],
        ['Chem_exam', 'final_score'],
        ['R-Bench', 'accuracy'],
        ['kcle', 'accuracy'],
        ['kcle_fix', 'accuracy'],
        ['cardbiomedbench', 'accuracy'],
        ['medmcqa', 'accuracy'],
        ['OpenSWI-deep-1k', 'score'],
        ['OpenSWI-shallow-1k', 'score'],
        ['CMPhysBench-fix_prompt_repeat8', 'score (8 runs average)'],
        ['atlas-val', 'accuracy (4 runs average)'],
        ['bio_data', 'naive_average'],
        ['MolecularIQ', 'naive_average'],
        ['mol_instruct', 'naive_average'],
        ['smolinstruct', 'naive_average'],
        ['smolinstruct_mini', 'naive_average'],
        ['matbench', 'naive_average'],
        '',
        ['srbench', 'mean_RMSE'],
        ['srbench', 'mean_NMSE'],
        ['srbench', 'mean_R2'],
        ['srbench', 'SymbolicMatch'],
        '',
        ['OlympiadBenchMath', 'accuracy'],
        ['OlympiadBenchPhysics', 'accuracy'],
        '',
        'ChatML Datasets',
        ['C-MHChem', 'accuracy'],
        ['CPsyExam', 'accuracy'],
        ['MaScQA', 'accuracy'],
        ['CS-Bench_CN_test', 'accuracy'],
        ['CS-Bench_EN_test', 'accuracy'],
        ['CS-Bench_CN_valid', 'accuracy'],
        ['CS-Bench_EN_valid', 'accuracy'],
        ['UGPhysics', 'naive_average'],
        ['HMMT2025_repeat32', 'accuracy (32 runs average)'],
        ['IMO-Bench-AnswerBench', 'accuracy'],
        ['AMO-Bench', 'accuracy'],
        ['UGD_hard_repeat8', 'accuracy (8 runs average)'],
        '',
        'CompassAcademic Extended',
        ['aime2025_repeat_32_CompassAcademic', 'accuracy (32 runs average)'],
        ['GPQA_diamond_repeat_4_CompassAcademic', 'accuracy (4 runs average)'],
        ['hle_llmjudge_CompassAcademic', 'accuracy'],
        [
            'lcb_code_generation_repeat_6_CompassAcademic',
            'pass@1 (6 runs average)'
        ],
    ],
    summary_groups=summary_groups,
)

obj_llm_judge_cfg = models[0]

for item in datasets:
    try:
        if 'atlas' in item['abbr'] and 'judge_cfg' in item['eval_cfg'][
                'evaluator']:
            item['eval_cfg']['evaluator']['judge_cfg'] = dict(
                judgers=[obj_llm_judge_cfg])
        elif 'judge_cfg' in item['eval_cfg']['evaluator']:
            item['eval_cfg']['evaluator']['judge_cfg'] = obj_llm_judge_cfg
        elif 'judge_cfg' in item['eval_cfg']['evaluator']['llm_evaluator']:
            item['eval_cfg']['evaluator']['llm_evaluator'][
                'judge_cfg'] = obj_llm_judge_cfg
    except Exception:
        pass

for item in chatml_datasets:
    if item['evaluator']['type'] == 'llm_evaluator':
        item['evaluator']['judge_cfg'] = obj_llm_judge_cfg
    if item['evaluator']['type'] == 'cascade_evaluator':
        item['evaluator']['llm_evaluator']['judge_cfg'] = obj_llm_judge_cfg
