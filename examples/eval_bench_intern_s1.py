# flake8: noqa

from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask


#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    # Datasets
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f import aime2025_datasets
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_gen_772ea0 import (
        gpqa_datasets,
    )
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_nocot_genericllmeval_gen_08c1de import (
        mmlu_pro_datasets,
    )
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import (
        ifeval_datasets,
    )
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_0shot_instruct_gen import (
        smolinstruct_datasets_0shot_instruct as smolinstruct_datasets,
    )
    from opencompass.configs.datasets.ChemBench.ChemBench_llmjudge_gen_c584cf import (
        chembench_datasets,
    )
    from opencompass.configs.datasets.matbench.matbench_llm_judge_gen_0e9276 import (
        matbench_datasets,
    )
    from opencompass.configs.datasets.ProteinLMBench.ProteinLMBench_llmjudge_gen_a67965 import (
        proteinlmbench_datasets,
    )

    # Summary Groups
    from opencompass.configs.summarizers.groups.mmlu_pro import (
        mmlu_pro_summary_groups,
    )

    # Models
    from opencompass.configs.models.interns1.intern_s1 import \
        models as interns1_model

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
# datasets list for evaluation
# Only take LCB generation for evaluation

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')),
               [])

# LLM judge config: using LLM to evaluate predictions
judge_cfg = dict()

for item in datasets:
    item['infer_cfg']['inferencer']['max_out_len'] = 65536
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg
    if 'llm_evaluator' in item['eval_cfg']['evaluator'].keys() and 'judge_cfg' in item['eval_cfg']['evaluator']['llm_evaluator']:
        item['eval_cfg']['evaluator']['llm_evaluator']['judge_cfg'] = judge_cfg


#######################################################################
#                       PART 2  Datset Summarizer                     #
#######################################################################

summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)

summary_groups.extend(
    [
        {
            'name': 'ChemBench',
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
    ]
)

summarizer = dict(
    dataset_abbrs=[
        'Knowledge',
        ['mmlu_pro', 'accuracy'],
        '',
        'Instruction Following',
        ['IFEval', 'Prompt-level-strict-accuracy'],
        '',
        'General Reasoning',
        ['GPQA_diamond', 'accuracy'],
        '',
        'Math Calculation',
        ['aime2025', 'accuracy'],
        '',
        'Academic',
        ['ChemBench', 'naive_average'],
        ['ProteinLMBench', 'accuracy'],
        '',
        'SmolInstruct',
        ['NC-I2F-0shot-instruct', 'score'],
        ['NC-I2S-0shot-instruct', 'score'],
        ['NC-S2F-0shot-instruct', 'score'],
        ['NC-S2I-0shot-instruct', 'score'],
        ['PP-ESOL-0shot-instruct', 'score'],
        ['PP-Lipo-0shot-instruct', 'score'],
        ['PP-BBBP-0shot-instruct', 'accuracy'],
        ['PP-ClinTox-0shot-instruct', 'accuracy'],
        ['PP-HIV-0shot-instruct', 'accuracy'],
        ['PP-SIDER-0shot-instruct', 'accuracy'],
        ['MC-0shot-instruct', 'score'],
        ['MG-0shot-instruct', 'score'],
        ['FS-0shot-instruct', 'score'],
        ['RS-0shot-instruct', 'score'],
        '',
        ['matbench_expt_gap', 'mae'],
        ['matbench_steels', 'mae'],
        ['matbench_expt_is_metal', 'accuracy'],
        ['matbench_glass', 'accuracy'],
        '',
    ],
    summary_groups=summary_groups,
)

#######################################################################
#                        PART 3  Models  List                         #
#######################################################################

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

#######################################################################
#                 PART 4  Inference/Evaluation Configuaration         #
#######################################################################

# infer with local runner
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=0,  # Modify if needed
        task=dict(type=OpenICLInferTask),
    ),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLEvalTask)),
)

#######################################################################
#                      PART 5  Utils Configuaration                   #
#######################################################################

work_dir = './outputs/oc_bench_intern_s1'
