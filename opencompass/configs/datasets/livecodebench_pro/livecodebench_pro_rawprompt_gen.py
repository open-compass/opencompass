from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    LCBProDataset,
    LCBProEvaluator,
)

lcb_pro_reader_cfg = dict(
    input_columns=[
        'problem',
    ],
    output_column='id_ddm',
)

lcb_pro_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{problem}'}
        ]
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

lcb_pro_eval_cfg = dict(
    evaluator=dict(
        type=LCBProEvaluator,
        submit_url='http://lightcpverifier.ailab.ailab.ai/submit',
        result_url='http://lightcpverifier.ailab.ailab.ai/result/{submission_id}',
        timeout=10,
        poll_interval=10,
        max_retries=3,
    )
)

lcb_pro_datasets = [
    dict(
        type=LCBProDataset,
        abbr='lcb_pro',
        path='opencompass/lcb_pro',
        reader_cfg=lcb_pro_reader_cfg,
        infer_cfg=lcb_pro_infer_cfg,
        eval_cfg=lcb_pro_eval_cfg,
    )
]
