from opencompass.datasets import CustomDataset
from opencompass.evaluator import MATHVerifyEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

aime2026_reader_cfg = dict(input_columns=['problem'], output_column='answer')

aime2026_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {
                'role': 'user',
                'content': '{problem}\nRemember to put your final answer within \\boxed{}.',
            },
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

aime2026_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)

aime2026_datasets = [
    dict(
        type=CustomDataset,
        abbr='aime2026',
        path='opencompass/aime2026',
        reader_cfg=aime2026_reader_cfg,
        infer_cfg=aime2026_infer_cfg,
        eval_cfg=aime2026_eval_cfg,
        n=1,
    )
]
