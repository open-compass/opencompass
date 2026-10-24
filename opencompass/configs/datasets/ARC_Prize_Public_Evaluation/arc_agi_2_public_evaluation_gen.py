from opencompass.datasets.arc_prize_public_evaluation import (
    ARCPrizeDataset,
    ARCPrizeEvaluator,
    ARCPrizeGenInferencer,
)
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

arc_agi_2_public_evaluation_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='output_test_data',
)

arc_agi_2_public_evaluation_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                # ARC Prize benchmarking sends the complete owner prompt as a
                # single user message for both ARC-AGI-1 and ARC-AGI-2.
                dict(role='HUMAN', prompt='{prompt}'),
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=ARCPrizeGenInferencer,
        enable_second_pass=True,
        second_pass_max_out_len=4096,
    ),
)

arc_agi_2_public_evaluation_eval_cfg = dict(
    evaluator=dict(
        type=ARCPrizeEvaluator,
        protocol='owner',
        # ARC-AGI-2 accepts two candidate outputs for each test input. A pair
        # is correct when either candidate exactly matches the gold output.
        attempt_aggregation='any',
        attempts_per_pair=2,
    ), )

arc_agi_2_public_evaluation_datasets = [
    dict(
        abbr='ARC_AGI_2_Public_Evaluation',
        type=ARCPrizeDataset,
        protocol='owner',
        version='arc_agi_2',
        path='opencompass/arc_agi_2_public_evaluation',
        reader_cfg=arc_agi_2_public_evaluation_reader_cfg,
        infer_cfg=arc_agi_2_public_evaluation_infer_cfg,
        eval_cfg=arc_agi_2_public_evaluation_eval_cfg,
        n=2,
        k=2,
    ),
]
