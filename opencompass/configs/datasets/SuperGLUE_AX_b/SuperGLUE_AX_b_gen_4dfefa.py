from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import AXDatasetV2
from opencompass.utils.text_postprocessors import first_option_postprocess

AX_b_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',
)

AX_b_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                '{sentence1}\n{sentence2}\nIs the sentence below entailed by the sentence above?\nA. Yes\nB. No\nAnswer:'
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

AX_b_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

AX_b_datasets = [
    dict(
        abbr='AX_b',
        type=AXDatasetV2,
        path='./data/SuperGLUE/AX-b/AX-b.jsonl',
        reader_cfg=AX_b_reader_cfg,
        infer_cfg=AX_b_infer_cfg,
        eval_cfg=AX_b_eval_cfg,
    )
]
