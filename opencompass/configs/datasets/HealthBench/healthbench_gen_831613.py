from opencompass.datasets import HealthBenchDataset, HealthBenchEvaluator
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


reader_cfg = dict(
    input_columns=[
        'prompt_trans',
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
                    prompt='{prompt_trans}', # prompt mode: zero-shot
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ChatInferencer),
)

# Evaluation configuration

healthbench_dataset = dict(
    type=HealthBenchDataset,
    abbr='healthbench',
    path='huihuixu/healthbench',
    subset='',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=dict(
        evaluator=dict(type=HealthBenchEvaluator, n_repeats=1, n_threads=1, subset_name=''),
        pred_role='BOT',
    ),
)
healthbench_hard_dataset = dict(
    type=HealthBenchDataset,
    abbr='healthbench_hard',
    path='huihuixu/healthbench',
    subset='hard',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=dict(
        evaluator=dict(type=HealthBenchEvaluator, n_repeats=1, n_threads=1, subset_name='hard'),
        pred_role='BOT',
    ),
)
healthbench_consensus_dataset = dict(
    type=HealthBenchDataset,
    abbr='healthbench_consensus',
    path='huihuixu/healthbench',
    subset='consensus',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=dict(
        evaluator=dict(type=HealthBenchEvaluator, n_repeats=1, n_threads=1, subset_name='consensus'),
        pred_role='BOT',
    ),
)

healthbench_datasets = [healthbench_dataset, healthbench_hard_dataset, healthbench_consensus_dataset]