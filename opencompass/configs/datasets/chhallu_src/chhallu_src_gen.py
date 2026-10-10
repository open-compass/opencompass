"""CHHallu-Src generation-based eval config.

Place at: opencompass/configs/datasets/chhallu_src/chhallu_src_gen.py
Use in a run config:  from opencompass.configs.datasets.chhallu_src.chhallu_src_gen import chhallu_src_datasets
"""
from opencompass.datasets import CHHalluSrcDataset, CHHalluSrcEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

chhallu_src_reader_cfg = dict(input_columns=["prompt"], output_column="row")

chhallu_src_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role="HUMAN", prompt="{prompt}")]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=256),
)

chhallu_src_eval_cfg = dict(
    evaluator=dict(type=CHHalluSrcEvaluator),
    pred_role="BOT",
)

chhallu_src_datasets = [
    dict(
        type=CHHalluSrcDataset,
        abbr="chhallu-src-v1",
        path="lizhuojun/chhallu-src-v1",
        reader_cfg=chhallu_src_reader_cfg,
        infer_cfg=chhallu_src_infer_cfg,
        eval_cfg=chhallu_src_eval_cfg,
    )
]
