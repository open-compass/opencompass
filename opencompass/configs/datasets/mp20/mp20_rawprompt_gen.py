from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.datasets import MP20Dataset, MP20Evaluator

mp20_reader_cfg = dict(
    input_columns=['system_prompt', 'user_input'],
    output_column='ground_truth',
)

mp20_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'system', 'content': '{system_prompt}'},
            {'role': 'user', 'content': '{user_input}'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mp20_eval_cfg = dict(
    evaluator=dict(type=MP20Evaluator),
    pred_role='BOT',
)

mp20_datasets = [
    dict(
        abbr='mp20',
        type=MP20Dataset,
        path='opencompass/mp20',
        reader_cfg=mp20_reader_cfg,
        infer_cfg=mp20_infer_cfg,
        eval_cfg=mp20_eval_cfg,
    )
]
