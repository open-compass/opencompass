from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import AXDatasetV2
from opencompass.utils.text_postprocessors import first_option_postprocess

RTE_reader_cfg = dict(
    input_columns=['hypothesis', 'premise'],
    output_column='label',
)

RTE_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                '{premise}\n{hypothesis}\nIs the sentence below entailed by the sentence above?\nA. Yes\nB. No\nAnswer:'
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

RTE_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

RTE_datasets = [
    dict(
        abbr='RTE',
        type=AXDatasetV2,  # rte share the same format with ax
        path='./data/SuperGLUE/RTE/val.jsonl',
        reader_cfg=RTE_reader_cfg,
        infer_cfg=RTE_infer_cfg,
        eval_cfg=RTE_eval_cfg,
    )
]
