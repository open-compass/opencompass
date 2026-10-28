from mmengine.config import read_base
import copy as _copy

with read_base():
    # CI runtime configuration. Keep these imports in the same read_base block
    # as the datasets so MMEngine resolves model configs eagerly.
    from autotest.all.config import common_eval as eval  # noqa: F401
    from autotest.all.config import common_infer as infer  # noqa: F401
    from autotest.all.config import raw_template_models as models  # noqa: F401
    from autotest.all.config import MOCK_API_MAX_WORKERS, set_mock_test_range

    # Datasets
    # Instruct Following
    from opencompass.configs.datasets.IFEval.IFEval_rawprompt_gen_e7f781 import (
        ifeval_datasets, )
    from opencompass.configs.datasets.IFBench.IFBench_rawprompt_gen import (
        ifbench_datasets, )
    from opencompass.configs.datasets.advancedIF.advancedIF_rawprompt_gen import (
        advancedif_datasets, )

    # General Reasoning
    from opencompass.configs.datasets.bbeh.bbeh_llmjudge_rawprompt_gen_36b5f4 import (
        bbeh_datasets, )
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_rawprompt_gen_706039 import (
        gpqa_datasets, )
    from opencompass.configs.datasets.korbench.korbench_single_0shot_cascade_eval_rawprompt_gen_c048da import (
        korbench_0shot_single_datasets)
    from opencompass.configs.datasets.supergpqa.supergpqa_cascade_rawprompt_gen_ca8345 import (
        supergpqa_datasets, )
    from opencompass.configs.datasets.HLE.hle_llmverify_rawprompt_gen_0970dd import (
        hle_datasets, )
    from opencompass.configs.datasets.HLE.hle_verified_rawprompt import (
        hle_datasets as hle_verified_datasets, )

    # Math Calculation
    from opencompass.configs.datasets.aime2024.aime2024_cascade_eval_rawprompt_gen_2f2c96 import (
        aime2024_datasets)
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_rawprompt_gen_2f2c96 import (
        aime2025_datasets)
    from opencompass.configs.datasets.aime2026.aime2026_cascade_eval_rawprompt_gen_0970dd import (
        aime2026_datasets)
    from opencompass.configs.datasets.hmmt2026.hmmt2026_cascade_eval_rawprompt_gen_0970dd import (
        hmmt2026_datasets)
    from opencompass.configs.datasets.OlympiadBench.OlympiadBench_0shot_llmverify_rawprompt_gen_d3e9e4 import (
        olympiadbench_datasets, )
    from opencompass.configs.datasets.livemathbench.livemathbench_hard_custom_cascade_eval_rawprompt_gen_e1ce64 import (
        livemathbench_datasets, )
    from opencompass.configs.datasets.OlymMATH.olymmath_llmverify_rawprompt_gen_9d3a8e import (
        olymmath_datasets, )
    from opencompass.configs.datasets.math.math_500_cascade_eval_rawprompt_gen_0970dd import (
        math_datasets, )

    # Knowledge
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_nocot_genericllmeval_rawprompt_gen_0321fb import (
        mmlu_pro_datasets, )
    from opencompass.configs.datasets.cmmlu.cmmlu_llmjudge_rawprompt_gen_9f9c31 import (
        cmmlu_datasets, )
    from opencompass.configs.datasets.mmlu.mmlu_llmjudge_rawprompt_gen_af67f0 import (
        mmlu_datasets, )
    from opencompass.configs.datasets.SimpleQA.simpleqa_verified_rawprompt_gen import (
        simpleqa_verified_datasets, )
    from opencompass.configs.datasets.drop.drop_0shot_llmjudge_rawprompt_gen import (
        drop_datasets as drop_0shot_datasets, )
    from opencompass.configs.datasets.drop.drop_3shot_llmjudge_rawprompt_gen import (
        drop_datasets as drop_3shot_datasets, )

    # Coding
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_rawprompt_gen_6ce2ca import (
        humaneval_datasets, )
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_0shot_nocot_rawprompt_gen_30c1e5 import (
        sanitized_mbpp_datasets, )
    from opencompass.configs.datasets.livecodebench.livecodebench_rawprompt_gen_c09673 import (
        LCBCodeGeneration_dataset, )

    # AI4S
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_0shot_instruct_rawprompt_gen import (
        smolinstruct_datasets_0shot_instruct as smolinstruct_datasets,
        mini_smolinstruct_datasets_0shot_instruct as
        mini_smolinstruct_datasets,
    )
    from opencompass.configs.datasets.ChemBench.ChemBench_llmjudge_rawprompt_gen_fa3fc4 import (
        chembench_datasets, )
    from opencompass.configs.datasets.MedXpertQA.MedXpertQA_llmjudge_rawprompt_gen import (
        medxpertqa_datasets, )
    from opencompass.configs.datasets.PHYSICS.PHYSICS_llm_judge_rawprompt_gen_56ebc8 import (
        physics_datasets, )
    from opencompass.configs.datasets.ClimaQA.ClimaQA_Gold_llm_judge_rawprompt_gen_b3080f import (
        climaqa_datasets, )
    from opencompass.configs.datasets.matbench.matbench_llm_judge_rawprompt_gen_c987b6 import (
        matbench_datasets, )
    from opencompass.configs.datasets.ProteinLMBench.ProteinLMBench_llmjudge_rawprompt_gen_9627a6 import (
        proteinlmbench_datasets, )
    from opencompass.configs.datasets.srbench.srbench_rawprompt_gen import (
        srbench_datasets, )
    from opencompass.configs.datasets.PHYBench.phybench_rawprompt_gen import (
        phybench_datasets, )
    from opencompass.configs.datasets.Earth_Silver.Earth_Silver_llmjudge_rawprompt_gen_a84bc6 import (
        earth_silver_mcq_datasets, )
    from opencompass.configs.datasets.chem_exam.competition_rawprompt_gen import (
        chem_competition_instruct_datasets, )
    from opencompass.configs.datasets.chem_exam.gaokao_rawprompt_gen import (
        chem_gaokao_instruct_datasets, )
    from opencompass.configs.datasets.R_Bench.rbench_llmjudge_rawprompt_gen_c24221 import (
        RBench_datasets, )
    from opencompass.configs.datasets.kcle.kcle_llm_judge_rawprompt_gen_16e383 import (
        kcle_datasets as kcle_fix_datasets)
    from opencompass.configs.datasets.CARDBiomedBench.CARDBiomedBench_llmjudge_rawprompt_gen_b4d90c import (
        cardbiomedbench_datasets, )
    from opencompass.configs.datasets.medmcqa.medmcqa_llmjudge_rawprompt_gen_015178 import (
        medmcqa_datasets)
    from opencompass.configs.datasets.openswi.openswi_rawprompt_gen import (
        openswi_datasets)
    from opencompass.configs.datasets.CMPhysBench.cmphysbench_rawprompt_gen import (
        cmphysbench_datasets)
    from opencompass.configs.datasets.biodata.biodata_task_rawprompt_gen import (
        biodata_task_datasets,
        mini_biodata_task_datasets,
    )
    from opencompass.configs.datasets.MolInstructions_chem.mol_instructions_chem_rawprompt_gen import (
        mol_gen_selfies_datasets)
    from opencompass.configs.datasets.atlas.atlas_val_rawprompt_gen_277bee import (
        atlas_datasets)
    from opencompass.configs.datasets.molculariq.molculariq_rawprompt_gen import (
        moleculariq_datasets)
    from opencompass.configs.datasets.s2_tomg_bench.s2_tomg_bench_gen import (
        mini_s2_tomg_bench_datasets)
    from opencompass.configs.datasets.scireasoner1_5.scireasoner1_5_llm_extract import (
        mini_scireasoner1_5_datasets)
    from opencompass.configs.datasets.mp20.mp20_rawprompt_gen import (
        mp20_datasets)
    from opencompass.configs.datasets.CLBench.clbench_llmjudge_rawprompt_gen_84a8f0 import (
        clbench_datasets)
    from opencompass.configs.datasets.eese.eese_llm_judge_rawprompt_gen import (
        eese_datasets)
    from opencompass.configs.datasets.ZebraLogic.zebralogic_rawprompt_gen_9446ef import (
        zebralogic_datasets)

    # ChatML Datasets
    from opencompass.configs.chatml_datasets.IMO_Bench_AnswerBench.IMO_Bench_AnswerBench_gen import datasets as IMO_Bench_AnswerBench_chatml
    from opencompass.configs.chatml_datasets.UGD_hard.UGD_hard_repeat8_gen import datasets as UGD_hard_chatml

    # Summary Groups
    from opencompass.configs.summarizers.groups.bbeh import bbeh_summary_groups
    from opencompass.configs.summarizers.groups.mmlu_pro import (
        mmlu_pro_summary_groups, )
    from opencompass.configs.summarizers.groups.cmmlu import (
        cmmlu_summary_groups, )
    from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups
    from opencompass.configs.summarizers.groups.korbench import (
        korbench_summary_groups, )

    from opencompass.configs.summarizers.groups.OlympiadBench import (
        OlympiadBench_summary_groups, )
    from opencompass.configs.summarizers.groups.OlympiadBench import (
        OlympiadBenchMath_summary_groups,
        OlympiadBenchPhysics_summary_groups,
    )
    from opencompass.configs.summarizers.groups.PHYSICS import (
        physics_summary_groups, )
    from opencompass.configs.summarizers.groups.supergpqa import (
        supergpqa_summary_groups, )
    from opencompass.configs.summarizers.groups.biodata import (
        biodata_summary_groups, )
    from opencompass.configs.summarizers.groups.biodata import (
        mini_biodata_summary_groups, )
    from opencompass.configs.summarizers.groups.matbench import (
        matbench_summary_groups, )
    from opencompass.configs.summarizers.groups.mol_instruct import (
        mol_instruct_summary_groups, )
    from opencompass.configs.summarizers.groups.smolinstruct import (
        smolinstruct_summary_groups, smolinstruct_mini_summary_groups)
    from opencompass.configs.summarizers.groups.scireasoner1_5 import (
        scireasoner1_5_mini_summary_groups)
    from opencompass.configs.summarizers.groups.s2_tomg_bench import (
        s2_tomg_bench_summary_groups)

# Each of the three mock suites runs in its own process against one API server.
# Bound each process so their combined request load remains below the server
# concurrency limit.
for model in models:
    model['max_workers'] = min(
        int(model.get('max_workers') or MOCK_API_MAX_WORKERS),
        MOCK_API_MAX_WORKERS,
    )

# Add latest LCB versions.
LCBCodeGeneration_v6_datasets = _copy.deepcopy(LCBCodeGeneration_dataset)
LCBCodeGeneration_v6_datasets["abbr"] = "lcb_code_generation_v6"
LCBCodeGeneration_v6_datasets["release_version"] = "v6"
LCBCodeGeneration_v6_datasets["eval_cfg"]["evaluator"][
    "release_version"] = "v6"
LCBCodeGeneration_v6_datasets["eval_cfg"]["evaluator"][
    "extractor_version"] = "v2"
LCBCodeGeneration_v6_datasets = [LCBCodeGeneration_v6_datasets]

LCBCodeGeneration_v5_datasets = _copy.deepcopy(LCBCodeGeneration_dataset)
LCBCodeGeneration_v5_datasets["abbr"] = "lcb_code_generation_v5"
LCBCodeGeneration_v5_datasets["release_version"] = "v5"
LCBCodeGeneration_v5_datasets["eval_cfg"]["evaluator"][
    "release_version"] = "v5"
LCBCodeGeneration_v5_datasets["eval_cfg"]["evaluator"][
    "extractor_version"] = "v2"
LCBCodeGeneration_v5_datasets = [LCBCodeGeneration_v5_datasets]

math_datasets[0]['abbr'] = 'math500_prm800k'

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)

chatml_datasets = sum(
    (v for k, v in locals().items() if k.endswith('_chatml')),
    [],
)

summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], [])

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
            'ChemBench_Name_Conversion',
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
        'name': 'ClimaQA',
        'subsets':
        ['ClimaQA_Gold_cloze', 'ClimaQA_Gold_ffq', 'ClimaQA_Gold_mcq'],
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
            'livemathbench_hard_custom_hard_en'
        ]
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
        'name': 'AdvancedIF',
        'subsets': [
            'advancedIF_CC',
            'advancedIF_CIF',
            'advancedIF_SS',
        ],
    },
    {
        'name': 'Zebralogic',
        'subsets': [
            'zebralogic_mc',
            'zebralogic_grid',
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
        ['drop_0shot', 'accuracy'],
        ['drop_3shot', 'accuracy'],
        '',
        'Instruction Following',
        ['IFEval', 'Prompt-level-strict-accuracy'],
        ['IFBench', 'score'],
        ['AdvancedIF', 'naive_average'],
        '',
        'General Reasoning',
        ['bbeh', 'naive_average'],
        ['bbeh', 'harmonic_mean'],
        ['GPQA_diamond', 'accuracy'],
        ['GPQA_diamond', 'G-Pass@8_0.0'],
        ['korbench_single', 'accuracy'],
        ['korbench_single', 'G-Pass@4_0.0'],
        ['SuperGPQA', 'weighted_average'],
        ['hle_llmjudge', 'accuracy'],
        ['hle_verified_llmjudge', 'accuracy'],
        ['OlympiadBench', 'accuracy'],
        '',
        'Math Calculation',
        ['aime2024', 'accuracy'],
        ['aime2024', 'G-Pass@32_0.0'],
        ['aime2025', 'accuracy'],
        ['aime2025', 'G-Pass@32_0.0'],
        ['aime2026', 'accuracy'],
        ['aime2026', 'G-Pass@32_0.0'],
        ['hmmt2026', 'accuracy'],
        ['hmmt2026', 'G-Pass@32_0.0'],
        ['livemathbench_hard', 'accuracy'],
        ['livemathbench_hard', 'G-Pass@32_0.0'],
        ['olymmath_llmjudge', 'naive_average'],
        ['math500_prm800k', 'accuracy'],
        ['math500_prm800k', 'G-Pass@4_0.0'],
        '',
        'Code',
        ['openai_humaneval', 'humaneval_pass@1'],
        ['sanitized_mbpp', 'score'],
        ['lcb_code_generation_v5', 'pass@1'],
        ['lcb_code_generation_v6', 'pass@1'],
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
        ['CMPhysBench-fix_prompt', 'score'],
        ['atlas-val', 'accuracy'],
        ['bio_data', 'naive_average'],
        ['mini_bio_data', 'naive_average'],
        ['MolecularIQ', 'naive_average'],
        ['mol_instruct', 'naive_average'],
        ['smolinstruct', 'naive_average'],
        ['smolinstruct_mini', 'naive_average'],
        ['matbench', 'naive_average'],
        ['S2-TOMG-Bench-mini', 'naive_average'],
        ['mp20', 'match_rate'],
        ['clbench', 'accuracy'],
        ['eese-llmjudge', 'overall_score'],
        ['Zebralogic', 'naive_average'],
        '',
        ['SciReasoner1_5-mini', 'naive_average'],
        ['SciReasoner1_5-OQMD-mini', 'naive_average'],
        ['SciReasoner1_5-JARVISDFT-mini', 'naive_average'],
        ['SciReasoner1_5-GO-BP-mini', 'naive_average'],
        ['SciReasoner1_5-TMScore-MAE-mini', 'naive_average'],
        ['SciReasoner1_5-DUDE-AUC-mini', 'naive_average'],
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
        ['IMO-Bench-AnswerBench', 'accuracy'],
        ['UGD_hard', 'accuracy'],
    ],
    summary_groups=summary_groups,
)

# CI runtime configuration. Dataset selection and summarization above mirror
# the fullbench_v2_0 objective settings; only the mock-compatible runner and
# model definitions are supplied by the imports at the top of this file.
obj_llm_judge_cfg = models[0]
for item in datasets:
    try:
        evaluator = item['eval_cfg']['evaluator']
        if 'atlas' in item['abbr'] and 'judge_cfg' in evaluator:
            evaluator['judge_cfg'] = dict(judgers=[obj_llm_judge_cfg])
        elif 'judge_cfg' in evaluator:
            evaluator['judge_cfg'] = obj_llm_judge_cfg
        elif 'llm_evaluator' in evaluator and 'judge_cfg' in evaluator[
                'llm_evaluator']:
            evaluator['llm_evaluator']['judge_cfg'] = obj_llm_judge_cfg
    except (KeyError, TypeError):
        pass

for dataset in chatml_datasets:
    evaluator = dataset.get('evaluator', {})
    if evaluator.get('type') == 'llm_evaluator':
        evaluator['judge_cfg'] = obj_llm_judge_cfg
    elif evaluator.get('type') == 'cascade_evaluator':
        evaluator.setdefault('llm_evaluator',
                             {})['judge_cfg'] = obj_llm_judge_cfg

set_mock_test_range(datasets)
set_mock_test_range(chatml_datasets)
del set_mock_test_range

# Repeated evaluations require complete n-sized groups. A ten-sample mock
# slice cannot preserve the production repeat counts (4/8/32), so exercise
# each evaluator once in CI.
for dataset in datasets + chatml_datasets:
    if 'n' in dataset:
        dataset['n'] = 1
    if 'k' in dataset:
        dataset['k'] = 1
    if dataset.get('abbr') == 'openai_humaneval':
        dataset['eval_cfg']['evaluator']['ignore_incomplete'] = True
del dataset
