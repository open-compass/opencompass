from opencompass.datasets.arc_prize_public_evaluation import (
    ARCPrizeDataset,
    ARCPrizeEvaluator,
    ARCPrizeGenInferencer,
)
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

arc_prize_public_evaluation_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='output_test_data',
)

arc_prize_public_evaluation_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                # ARC Prize benchmarking sends the complete owner prompt as a
                # single user message. Avoid a benchmark-specific system
                # prompt because it changes the evaluation signature.
                dict(role='HUMAN', prompt='{prompt}'),
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    # Output length remains model-configurable, matching the ARC Prize harness.
    inferencer=dict(
        type=ARCPrizeGenInferencer,
        enable_second_pass=True,
        second_pass_max_out_len=4096,
    ),
)

arc_prize_public_evaluation_eval_cfg = dict(evaluator=dict(
    type=ARCPrizeEvaluator,
    protocol='owner',
), )

arc_prize_public_evaluation_datasets = [
    dict(
        abbr='ARC_Prize_Public_Evaluation',
        type=ARCPrizeDataset,
        protocol='owner',
        version='arc_agi_1',
        path='opencompass/arc_prize_public_evaluation',
        reader_cfg=arc_prize_public_evaluation_reader_cfg,
        infer_cfg=arc_prize_public_evaluation_infer_cfg,
        eval_cfg=arc_prize_public_evaluation_eval_cfg,
    ),
]
