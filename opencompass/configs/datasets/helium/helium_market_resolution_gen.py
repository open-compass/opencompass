from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HeliumMarketResolutionDataset
from opencompass.datasets.helium import HeliumMarketResolutionEvaluator

# Helium Market Resolution — 300 frozen option-chain prompts (IV, delta, MCQ).
# Dataset: HeliumTrades/helium-market-resolution-benchmark
# Methodology: https://heliumtrades.com/benchmarks/

helium_mr_reader_cfg = dict(
    input_columns=["prompt"],
    output_column="reference",
)

helium_mr_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role="user", content="{prompt}"),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=256),
)

helium_mr_eval_cfg = dict(
    evaluator=dict(type=HeliumMarketResolutionEvaluator),
    pred_role="BOT",
)

helium_market_resolution_datasets = [
    dict(
        type=HeliumMarketResolutionDataset,
        abbr="helium_mr",
        path="HeliumTrades/helium-market-resolution-benchmark",
        mini=False,
        reader_cfg=helium_mr_reader_cfg,
        infer_cfg=helium_mr_infer_cfg,
        eval_cfg=helium_mr_eval_cfg,
    ),
    dict(
        type=HeliumMarketResolutionDataset,
        abbr="helium_mr_mini",
        path="HeliumTrades/helium-market-resolution-benchmark",
        mini=True,
        reader_cfg=helium_mr_reader_cfg,
        infer_cfg=helium_mr_infer_cfg,
        eval_cfg=helium_mr_eval_cfg,
    ),
]
