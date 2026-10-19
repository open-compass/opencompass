from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import QuALITYDataset, QuALITYEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess

QuALITY_reader_cfg = dict(
    input_columns=['article', 'question', 'A', 'B', 'C', 'D'],
    output_column='gold_label',
)

QuALITY_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                'Read the article, and answer the question.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}'
            ),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

QuALITY_eval_cfg = dict(
    evaluator=dict(type=QuALITYEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    pred_role='BOT')

QuALITY_datasets = [
    dict(
        abbr='QuALITY',
        type=QuALITYDataset,
        path='./data/QuALITY/QuALITY.v1.0.1.htmlstripped.dev',
        reader_cfg=QuALITY_reader_cfg,
        infer_cfg=QuALITY_infer_cfg,
        eval_cfg=QuALITY_eval_cfg),
]
