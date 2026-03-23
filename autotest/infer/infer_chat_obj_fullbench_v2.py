import copy

from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferConcurrentTask

with read_base():
    # Datasets
    from opencompass.configs.chatml_datasets.AMO_Bench.AMO_Bench_gen import \
        datasets as AMO_Bench_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.C_MHChem.C_MHChem_gen import \
        datasets as C_MHChem_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.CPsyExam.CPsyExam_gen import \
        datasets as CPsyExam_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.CS_Bench.CS_Bench_gen import \
        datasets as CS_Bench_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.HMMT2025.HMMT2025_gen import \
        datasets as HMMT2025_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.IMO_Bench_AnswerBench.IMO_Bench_AnswerBench_gen import \
        datasets as IMO_Bench_AnswerBench_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.MaScQA.MaScQA_gen import \
        datasets as MaScQA_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.UGD_hard.UGD_hard_gen import \
        datasets as UGD_hard_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.UGPhysics.UGPhysics_gen import \
        datasets as UGPhysics_chatml  # noqa: F401, E501
    from opencompass.configs.datasets.aime2024.aime2024_cascade_eval_rawprompt_gen_2f2c96 import \
        aime2024_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_rawprompt_gen_2f2c96 import \
        aime2025_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.aime2025.aime2025_llmjudge_academic import \
        aime2025_datasets as \
        CompassAcademic_aime2025_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.aime2026.aime2026_cascade_eval_rawprompt_gen_0970dd import \
        aime2026_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.atlas.atlas_val_rawprompt_gen_277bee import \
        atlas_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bbeh.bbeh_llmjudge_rawprompt_gen_36b5f4 import \
        bbeh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_complete_rawprompt_gen_95140b import \
        bigcodebench_hard_complete_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_instruct_rawprompt_gen_5cbb9f import \
        bigcodebench_hard_instruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.biodata.biodata_task_rawprompt_gen import \
        biodata_task_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.CARDBiomedBench.CARDBiomedBench_llmjudge_rawprompt_gen_b4d90c import \
        cardbiomedbench_datasets  # noqa: F401, E501
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
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_academic import \
        gpqa_datasets as CompassAcademic_gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_rawprompt_gen_706039 import \
        gpqa_datasets
    from opencompass.configs.datasets.HLE.hle_llmverify_academic import \
        hle_datasets as CompassAcademic_hle_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.HLE.hle_llmverify_rawprompt_gen_0970dd import \
        hle_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hmmt2026.hmmt2026_cascade_eval_rawprompt_gen_0970dd import \
        hmmt2026_datasets  # noqa: F401, E501
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
    from opencompass.configs.datasets.livecodebench.livecodebench_v6_academic import \
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
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_nocot_genericllmeval_rawprompt_gen_0321fb import \
        mmlu_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MolInstructions_chem.mol_instructions_chem_rawprompt_gen import \
        mol_gen_selfies_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.OlymMATH.olymmath_llmverify_rawprompt_gen_9d3a8e import \
        olymmath_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.OlympiadBench.OlympiadBench_0shot_llmverify_rawprompt_gen_be8b13 import \
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
    # AI4S
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_0shot_instruct_rawprompt_gen import \
        smolinstruct_datasets_0shot_instruct as \
        smolinstruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.srbench.srbench_rawprompt_gen import \
        srbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.supergpqa.supergpqa_cascade_rawprompt_gen_ca8345 import \
        supergpqa_datasets  # noqa: F401, E501

LCBCodeGeneration_v6_datasets = copy.deepcopy(LCBCodeGeneration_dataset)
LCBCodeGeneration_v6_datasets['abbr'] = 'lcb_code_generation_v6'
LCBCodeGeneration_v6_datasets['release_version'] = 'v6'
LCBCodeGeneration_v6_datasets['eval_cfg']['evaluator'][
    'release_version'] = 'v6'
LCBCodeGeneration_v6_datasets = [LCBCodeGeneration_v6_datasets]

LCBCodeGeneration_v5_datasets = copy.deepcopy(LCBCodeGeneration_dataset)
LCBCodeGeneration_v5_datasets['abbr'] = 'lcb_code_generation_v5'
LCBCodeGeneration_v5_datasets['release_version'] = 'v5'
LCBCodeGeneration_v5_datasets['eval_cfg']['evaluator'][
    'release_version'] = 'v5'
LCBCodeGeneration_v5_datasets = [LCBCodeGeneration_v5_datasets]

CompassAcademic_LCBCodeGeneration_datasets = [
    copy.deepcopy(CompassAcademic_LCBCodeGeneration_dataset)
]

cmphysbench_datasets[0]['abbr'] = cmphysbench_datasets[0]['abbr'] + '_repeat_8'

# CompassAcademic Extended Process

compassacademic_dataset_list = [
    CompassAcademic_aime2025_datasets,
    CompassAcademic_gpqa_datasets,
    CompassAcademic_LCBCodeGeneration_datasets,
    CompassAcademic_hle_datasets,
]
for acadatasets in compassacademic_dataset_list:
    for acadataset in acadatasets:
        acadataset['abbr'] = 'CompassAcademic_' + acadataset['abbr']

repeated_info = [
    (math_datasets, 4),
    (gpqa_datasets, 8),
    (aime2024_datasets, 32),
    (aime2025_datasets, 32),
    (olympiadbench_datasets, 1),
    (livemathbench_datasets, 32),
    (olymmath_datasets, 4),
    (korbench_0shot_single_datasets, 4),
    (cmphysbench_datasets, 8),
    (aime2026_datasets, 32),
    (hmmt2026_datasets, 32),
]

for datasets_, num in repeated_info:
    for dataset_ in datasets_:
        dataset_['n'] = num
        dataset_['k'] = num

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)

chatml_datasets = sum(
    (v for k, v in locals().items() if k.endswith('_chatml')),
    [],
)

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        retry=0,
        task=dict(type=OpenICLInferConcurrentTask),
    ),
)
