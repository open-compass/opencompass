from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.datasets import (PerspectiveGapDataset,
                                  PerspectiveGapPromptWritingEvaluator)

perspectivegap_prompt_writing_reader_cfg = dict(
    input_columns=['input'],
    output_column='reference_data',
)

perspectivegap_prompt_writing_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role='user', content='{input}'),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

perspectivegap_prompt_writing_eval_cfg = dict(
    evaluator=dict(type=PerspectiveGapPromptWritingEvaluator),
    pred_role='BOT',
)

perspectivegap_prompt_writing_datasets = [
    dict(
        type=PerspectiveGapDataset,
        path='sun1245/PerspectiveGap',
        name='prompt_writing',
        abbr='perspectivegap-prompt_writing',
        reader_cfg=perspectivegap_prompt_writing_reader_cfg,
        infer_cfg=perspectivegap_prompt_writing_infer_cfg,
        eval_cfg=perspectivegap_prompt_writing_eval_cfg,
    )
]
