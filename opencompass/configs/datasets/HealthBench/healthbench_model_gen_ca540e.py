from opencompass.datasets import HealthBenchDataset, HealthBenchEvaluator
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


# Reader configuration
reader_cfg = dict(
    input_columns=[
        'prompt_trans'
    ],
    output_column='prompt_id', # useless
)

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}', # prompt mode: zero-shot
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ChatInferencer),
)

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