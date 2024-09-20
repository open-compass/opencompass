from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import COPADatasetV2
from opencompass.utils.text_postprocessors import first_option_postprocess

COPA_reader_cfg = dict(
    input_columns=['question', 'premise', 'choice1', 'choice2'],
    output_column='label',
)

COPA_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=
                    '{premise}\nQuestion: Which may be the {question}?\nA. {choice1}\nB. {choice2}\nAnswer:'
                ),
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

COPA_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

COPA_datasets = [
    dict(
        abbr='COPA',
        type=COPADatasetV2,
        path='./data/SuperGLUE/COPA/val.jsonl',
        reader_cfg=COPA_reader_cfg,
        infer_cfg=COPA_infer_cfg,
        eval_cfg=COPA_eval_cfg,
    )
]
