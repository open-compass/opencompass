from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.datasets import MP20Dataset, MP20Evaluator

mp20_reader_cfg = dict(
    input_columns=['system_prompt', 'user_input'],
    output_column='ground_truth',
)

mp20_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt='{system_prompt}'),
            ],
            round=[
                dict(role='HUMAN', prompt='{user_input}'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16000),
)

mp20_eval_cfg = dict(
    evaluator=dict(type=MP20Evaluator),
    pred_role='BOT',
)

mp20_datasets = [
    dict(
        abbr='mp20',
        type=MP20Dataset,
        path='data/mp20/mp20_test.jsonl',
        reader_cfg=mp20_reader_cfg,
        infer_cfg=mp20_infer_cfg,
        eval_cfg=mp20_eval_cfg,
    )
]
