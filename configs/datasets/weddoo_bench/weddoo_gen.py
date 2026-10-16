from opencompass.openicl import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import WeddooBenchDataset, WeddooBenchEvaluator

# WeddooBench 数据集读取配置
weddoo_bench_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

weddoo_bench_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
                dict(role='BOT', prompt=''),
            ]
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

weddoo_bench_eval_cfg = dict(
    evaluator=dict(type=WeddooBenchEvaluator),
)

weddoo_bench_datasets = [
    dict(
        type=WeddooBenchDataset,
        path='./data/weddoo_bench/weddoo_bench_v1.jsonl',
        reader_cfg=weddoo_bench_reader_cfg,
        infer_cfg=weddoo_bench_infer_cfg,
        eval_cfg=weddoo_bench_eval_cfg,
    )
]
