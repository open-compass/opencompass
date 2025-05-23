from opencompass.datasets import HealthBenchDataset, HealthBenchEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import HealthBenchTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


# Reader configuration
reader_cfg = dict(
    input_columns=[
        'example_tags', 'ideal_completions_data', 'prompt', 'prompt_id', 'rubrics', 'canary'
    ],
    output_column='prompt_id', # useless
)

# Inference configuration
infer_cfg = dict(
    prompt_template=dict(
        type=HealthBenchTemplate,
        key='prompt_trans',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# infer_cfg = dict(
#     prompt_template=dict(
#         type=PromptTemplate,
#         template=dict(
#             round=[
#                 dict(
#                     role='HUMAN',
#                     prompt='{prompt_id}', # prompt mode: zero-shot
#                 ),
#                 dict(
#                     role='BOT',
#                     prompt='{prompt_id}', # prompt mode: zero-shot
#                 ),
#             ],
#         ),
#     ),
#     retriever=dict(type=ZeroRetriever),
#     inferencer=dict(type=GenInferencer),
# )

# Evaluation configuration
eval_cfg = dict(
    evaluator=dict(type=HealthBenchEvaluator),
    pred_role='BOT',
)
healthbench_vanilla_dataset = dict(
    type=HealthBenchDataset,
    abbr='healthbench_vanilla',
    path='huihuixu/healthbench',
    subset='vanilla',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)
healthbench_hard_dataset = dict(
    type=HealthBenchDataset,
    abbr='healthbench_hard',
    path='huihuixu/healthbench',
    subset='hard',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)
healthbench_consensus_dataset = dict(
    type=HealthBenchDataset,
    abbr='healthbench_consensus',
    path='huihuixu/healthbench',
    subset='consensus',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)

healthbench_all_datasets = [healthbench_vanilla_dataset, healthbench_hard_dataset, healthbench_consensus_dataset ]