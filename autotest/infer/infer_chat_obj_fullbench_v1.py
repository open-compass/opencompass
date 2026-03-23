from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferConcurrentTask

with read_base():
    from autotest.infer.models import models  # noqa: F401, E501
    from opencompass.configs.datasets.aime2024.aime2024_cascade_eval_gen_5e9f4f import \
        aime2024_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.aime2024.aime2024_gen_6e39a4 import \
        aime2024_datasets as aime2024_gen_datasets
    from opencompass.configs.datasets.aime2024.aime2024_llmjudge_gen_5e9f4f import \
        aime2024_datasets as aime2024_llmjudge_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f import \
        aime2025_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.aime2025.aime2025_llmjudge_academic import \
        aime2025_datasets as \
        CompassAcademic_aime2025_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.aime2025.aime2025_llmjudge_gen_5e9f4f import \
        aime2025_datasets as aime2025_llmjudge_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.aime2026.aime2026_cascade_eval_gen_6ff468 import \
        aime2026_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import \
        ARC_c_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ARC_Prize_Public_Evaluation.arc_prize_public_evaluation_gen_872059 import \
        arc_prize_public_evaluation_datasets as \
        arc_prize_public_evaluation_872059_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ARC_Prize_Public_Evaluation.arc_prize_public_evaluation_gen_fedd04 import \
        arc_prize_public_evaluation_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.atlas.atlas_val_gen_b2d1b6 import \
        atlas_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bbeh.bbeh_llmjudge_gen_86c3a0 import \
        bbeh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import \
        bbh_datasets as bbh_gen_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bbh.bbh_llmjudge_gen_b5bdf1 import \
        bbh_datasets as bbh_llmjudge_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_complete_gen_2888d3 import \
        bigcodebench_hard_complete_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_complete_gen_faf748 import \
        bigcodebench_hard_complete_datasets as \
        bigcodebench_hard_complete_faf748_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_instruct_gen_8815eb import \
        bigcodebench_hard_instruct_datasets as \
        bigcodebench_hard_instruct_8815eb_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.bigcodebench.bigcodebench_hard_instruct_gen_c3d5ad import \
        bigcodebench_hard_instruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.biodata.biodata_task_gen import \
        biodata_task_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.CARDBiomedBench.CARDBiomedBench_llmjudge_gen_99a231 import \
        cardbiomedbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.chem_exam.competition_gen import \
        chem_competition_instruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.chem_exam.gaokao_gen import \
        chem_gaokao_instruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ChemBench.ChemBench_llmjudge_gen_c584cf import \
        chembench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ClimaQA.ClimaQA_Gold_llm_judge_gen_f15343 import \
        climaqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmmlu.cmmlu_0shot_cot_gen_305931 import \
        cmmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmmlu.cmmlu_llmjudge_gen_e1cd9a import \
        cmmlu_datasets as cmmlu_llmjudge_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmo_fib.cmo_fib_gen_2783e5 import \
        cmo_fib_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmo_fib.cmo_fib_gen_ace24b import \
        cmo_fib_datasets as cmo_fib_ace24b_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.CMPhysBench.cmphysbench_gen import \
        cmphysbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.drop.drop_llmjudge_gen_3857b0 import \
        drop_datasets as drop_llmjudge_datasets
    from opencompass.configs.datasets.drop.drop_openai_simple_evals_gen_3857b0 import \
        drop_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ds1000.ds1000_service_eval_gen_cbc84f import \
        ds1000_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.Earth_Silver.Earth_Silver_llmjudge_gen_46140c import \
        earth_silver_mcq_datasets  # noqa: F401, E501
    # from opencompass.configs.datasets.eese.eese_llm_judge_gen import \
    #    eese_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_4c31db import \
        GaokaoBench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_d16acb import \
        GaokaoBench_datasets as GaokaoBench_d16acb_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_0shot_nocot_genericllmeval_gen_772ea0 import \
        gpqa_datasets as gpqa_nocot_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_academic import \
        gpqa_datasets as CompassAcademic_gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_gen_772ea0 import \
        gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import \
        gpqa_datasets as gpqa_simple_evals_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_6e39a4 import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import \
        gsm8k_datasets as gsm8k_17d799_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import \
        hellaswag_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hellaswag.hellaswag_llmjudge_gen_809ef1 import \
        hellaswag_datasets as hellaswag_llmjudge_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.HLE.hle_llmverify_academic import \
        hle_datasets as CompassAcademic_hle_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.HLE.hle_llmverify_gen_6ff468 import \
        hle_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hmmt2026.hmmt2026_cascade_eval_gen_6ff468 import \
        hmmt2026_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_dcae0e import \
        humaneval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.humanevalx.humanevalx_gen_3d84a3 import \
        humanevalx_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFBench.IFBench_gen import \
        ifbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import \
        ifeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.kcle.kcle_llm_judge_gen import \
        kcle_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.kcle.kcle_llm_judge_gen_60327a import \
        kcle_datasets as kcle_fix_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.korbench.korbench_llmjudge_gen_56cf43 import \
        korbench_0shot_single_datasets as \
        korbench_llmjudge_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.korbench.korbench_single_0_shot_gen import \
        korbench_0shot_single_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.korbench.korbench_single_0shot_cascade_eval_gen_56cf43 import \
        korbench_0shot_single_datasets as \
        korbench_cascade_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import \
        LCBCodeGeneration_dataset  # noqa: F401, E501
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_b2b0fd import \
        LCB_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.livecodebench.livecodebench_v6_academic import \
        LCBCodeGeneration_dataset as \
        CompassAcademic_LCBCodeGeneration_dataset  # noqa: F401, E501
    from opencompass.configs.datasets.livecodebench_pro.livecodebench_pro_gen import \
        lcb_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.livemathbench.livemathbench_hard_custom_cascade_eval_gen_4bce59 import \
        livemathbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.matbench.matbench_llm_judge_gen_0e9276 import \
        matbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_0shot_gen_11c4b5 import \
        math_datasets as math_0shot_datasets
    from opencompass.configs.datasets.math.math_500_cascade_eval_gen_6ff468 import \
        math_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.math.math_500_llmjudge_gen_6ff468 import \
        math_datasets as math_llmjudge_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MathBench.mathbench_2024_gen_4b8f28 import \
        mathbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MathBench.mathbench_2024_gen_50a320 import \
        mathbench_datasets as mathbench_50a320_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import \
        sanitized_mbpp_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.medmcqa.medmcqa_llmjudge_gen_60c8f5 import \
        medmcqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MedXpertQA.MedXpertQA_llmjudge_gen import \
        medxpertqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_llmjudge_gen_f4336b import \
        mmlu_datasets as mmlu_llmjudge_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_openai_simple_evals_gen_b618ea import \
        mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import \
        mmlu_pro_datasets as mmlu_pro_cot_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_nocot_genericllmeval_gen_08c1de import \
        mmlu_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmmlu_lite.mmmlu_lite_gen_c51a84 import \
        mmmlu_lite_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MolInstructions_chem.mol_instructions_chem_gen import \
        mol_gen_selfies_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.musr.musr_gen_3622bb import \
        musr_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.musr.musr_llmjudge_gen_b47fd3 import \
        musr_datasets as musr_llmjudge_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.nq.nq_open_1shot_gen_2e45e5 import \
        nq_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.OlymMATH.olymmath_llmverify_gen_97b203 import \
        olymmath_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.OlympiadBench.OlympiadBench_0shot_llmverify_gen_be8b13 import \
        olympiadbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.openswi.openswi_gen import \
        openswi_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.PHYBench.phybench_gen import \
        phybench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.PHYSICS.PHYSICS_llm_judge_gen_a133a2 import \
        physics_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ProteinLMBench.ProteinLMBench_llmjudge_gen_a67965 import \
        proteinlmbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.R_Bench.rbench_llmjudge_gen_c89350 import \
        RBench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_cot_gen_d95929 import \
        race_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.scicode.scicode_gen_085b98 import \
        SciCode_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_0shot_instruct_gen import \
        smolinstruct_datasets_0shot_instruct as \
        smolinstruct_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.srbench.srbench_gen import \
        srbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_cot_gen_1d56df import \
        BoolQ_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.supergpqa.supergpqa_cascade_gen_1545c1 import \
        supergpqa_datasets as supergpqa_cascade_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.supergpqa.supergpqa_llmjudge_gen_12b8bc import \
        supergpqa_datasets as supergpqa_llmjudge_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.teval.teval_en_gen_1ac254 import \
        teval_datasets as teval_en_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.teval.teval_zh_gen_1ac254 import \
        teval_datasets as teval_zh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import \
        TheoremQA_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_bc5f21 import \
        triviaqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_c87d61 import \
        triviaqa_datasets as triviaqa_c87d61_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.wikibench.wikibench_gen_0978ad import \
        wikibench_datasets  # noqa: F401, E501

models = models

LCBCodeGeneration_v6_datasets = LCBCodeGeneration_dataset
LCBCodeGeneration_v6_datasets['abbr'] = 'lcb_code_generation_v6'
LCBCodeGeneration_v6_datasets['release_version'] = 'v6'
LCBCodeGeneration_v6_datasets['eval_cfg']['evaluator'][
    'release_version'] = 'v6'
LCBCodeGeneration_v6_datasets = [LCBCodeGeneration_v6_datasets]

CompassAcademic_LCBCodeGeneration_datasets = [
    CompassAcademic_LCBCodeGeneration_dataset
]

cmphysbench_datasets[0]['abbr'] = cmphysbench_datasets[0]['abbr'] + '_repeat_8'

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

gen_datasets = [
    *aime2024_gen_datasets,
    *bbh_gen_datasets,
]

for temp_dataset in gen_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_gen'

llmjudge_datasets = [
    *aime2024_llmjudge_datasets, *aime2025_llmjudge_datasets,
    *bbh_llmjudge_datasets, *cmmlu_llmjudge_datasets, *drop_llmjudge_datasets,
    *hellaswag_llmjudge_datasets, *korbench_llmjudge_datasets,
    *math_llmjudge_datasets, *mmlu_llmjudge_datasets,
    *supergpqa_llmjudge_datasets, *musr_llmjudge_datasets
]

for temp_dataset in llmjudge_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_llmjudge'

cascade_datasets = [*korbench_cascade_datasets, *supergpqa_cascade_datasets]

for temp_dataset in cascade_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_cascade'

for temp_dataset in arc_prize_public_evaluation_872059_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_872059'

for temp_dataset in bigcodebench_hard_complete_faf748_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_faf748'

for temp_dataset in bigcodebench_hard_instruct_8815eb_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_8815eb'

for temp_dataset in cmo_fib_ace24b_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_ace24b'

for temp_dataset in GaokaoBench_d16acb_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_d16acb'

for temp_dataset in gsm8k_17d799_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_17d799'

for temp_dataset in kcle_fix_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_fix'

for temp_dataset in mathbench_50a320_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_50a320'

for temp_dataset in gpqa_simple_evals_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_simple_evals'

for temp_dataset in mmlu_pro_cot_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_cot'

for temp_dataset in triviaqa_c87d61_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_c87d61'

for temp_dataset in math_0shot_datasets:
    temp_dataset['abbr'] = temp_dataset['abbr'] + '_0shot'

datasets = sum(
    (v for k, v in locals().items()
     if k.endswith('_datasets') and 'scicode' not in k.lower()
     and 'teval' not in k.lower() and 'dingo' not in k.lower()),
    [],
)

datasets += teval_en_datasets
datasets += teval_zh_datasets
datasets += SciCode_datasets

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        retry=0,
        task=dict(type=OpenICLInferConcurrentTask),
    ),
)
