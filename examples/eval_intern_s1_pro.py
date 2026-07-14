# flake8: noqa

from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from copy import deepcopy
from opencompass.utils.text_postprocessors import extract_non_reasoning_content
from opencompass.models import OpenAISDKStreaming

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    # Datasets

    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_nocot_genericllmeval_gen_08c1de import (
        mmlu_pro_datasets,
    )
    from opencompass.configs.datasets.gpqa.gpqa_cascade_eval_gen_772ea0 import (
        gpqa_datasets,
    )
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f import (
        aime2025_datasets,
    )
    from opencompass.configs.chatml_datasets.IMO_Bench_AnswerBench.IMO_Bench_AnswerBench_gen import (
        datasets as IMO_Bench_AnswerBench_chatml
    )
    from opencompass.configs.datasets.IFBench.IFBench_gen import (
        ifbench_datasets,
    )
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import (
        LCBCodeGeneration_dataset,
    )
    from opencompass.configs.datasets.SmolInstruct.smolinstruct_0shot_instruct_gen import (
        smolinstruct_datasets_0shot_instruct as smolinstruct_datasets,
    )
    from opencompass.configs.datasets.matbench.matbench_llm_judge_gen_0e9276 import (
        matbench_datasets,
    )
    from opencompass.configs.datasets.biodata.biodata_task_gen import (
        biodata_task_datasets
    )
    from opencompass.configs.datasets.MolInstructions_chem.mol_instructions_chem_gen import (
        mol_gen_selfies_datasets
    )

    # Summary Groups
    from opencompass.configs.summarizers.groups.mmlu_pro import \
        mmlu_pro_summary_groups
    from opencompass.configs.summarizers.groups.biodata import (
        biodata_summary_groups,
    )

LCBCodeGeneration_v6_datasets = deepcopy(LCBCodeGeneration_dataset)
LCBCodeGeneration_v6_datasets['abbr'] = 'lcb_code_generation_v6'
LCBCodeGeneration_v6_datasets['release_version'] = 'v6'
LCBCodeGeneration_v6_datasets['eval_cfg']['evaluator'][
    'release_version'
] = 'v6'
LCBCodeGeneration_v6_datasets = [LCBCodeGeneration_v6_datasets]

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
# datasets list for evaluation

repeated_info = [
    (gpqa_datasets, 8),
    (aime2025_datasets, 32),
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

# LLM judge config: using LLM to evaluate predictions
judge_cfg = dict(
    abbr='YOUR_JUDGE_MODEL',
    type=OpenAISDKStreaming,
    path='YOUR_JUDGE_MODEL',
    key='YOUR_JUDGE_KEY',
    openai_api_base='YOUR_JUDGE_URL',
    mode='mid',
    meta_template=dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
),
    query_per_second=16,
    batch_size=64,
    temperature=0.001,
    max_out_len=8192,
    max_seq_len=32768,
)

for item in datasets:
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = judge_cfg
    if 'llm_evaluator' in item['eval_cfg']['evaluator'].keys() and 'judge_cfg' in item['eval_cfg']['evaluator']['llm_evaluator']:
        item['eval_cfg']['evaluator']['llm_evaluator']['judge_cfg'] = judge_cfg

for item in chatml_datasets:
    if item['evaluator']['type'] == 'llm_evaluator':
        item['evaluator']['judge_cfg'] = judge_cfg
    if item['evaluator']['type'] == 'cascade_evaluator':
        item['evaluator']['llm_evaluator']['judge_cfg'] = judge_cfg


#######################################################################
#                       PART 2  Datset Summarizer                     #
#######################################################################

summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)

summarizer = dict(
    dataset_abbrs=[
        ['mmlu_pro', 'accuracy'],
        ['IFBench', 'score'],
        ['GPQA_diamond', 'accuracy (8 runs average)'],
        ['aime2025', 'accuracy (32 runs average)'],
        ['lcb_code_generation_v6', 'pass@1'],
        ['bio_data', 'naive_average'],
        ['IMO-Bench-AnswerBench', 'accuracy'],
        '',
        'Mol_Instruct',
        ['FS-selfies', 'score'],
        ['MC-selfies', 'score'],
        ['MG-selfies', 'score'],
        ['PP-selfies', 'score'],
        ['RP-selfies', 'score'],
        ['RS-selfies', 'score'],
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
    ],
    summary_groups=summary_groups,
)


#######################################################################
#                        PART 3  Models                               #
#######################################################################

api_meta_template = dict(
    round=[
        dict(role='SYSTEM', api_role='SYSTEM'), # System prompt is only needed when evaluating Bio_data and Mol_instructions
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

models = [
dict(
    abbr='intern-s1-pro',
    type=OpenAISDKStreaming,
    path='intern-s1-pro',
    key='YOUR_API_KEY',
    openai_api_base='YOUR_API_BASE',
    meta_template=api_meta_template,
    query_per_second=16,
    batch_size=8,
    temperature=0.8,
    retry=10,
    max_out_len=65536,
    max_seq_len=65536,
    extra_body={
            'chat_template_kwargs': {'enable_thinking': True} # Disable thinking when evaluating scientific benchmarks
            },
    pred_postprocessor=dict(
        type=extract_non_reasoning_content,
    ),
),
]

#######################################################################
#                 PART 4  Inference/Evaluation Configuaration         #
#######################################################################

# infer with local runner
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask),
    ),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner,
         max_num_workers=16,
        task=dict(type=OpenICLEvalTask)
    ),
)

#######################################################################
#                      PART 5  Utils Configuaration                   #
#######################################################################

work_dir = './outputs/oc_intern_s1_pro_eval'