from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CBDatasetV2
from opencompass.utils.text_postprocessors import first_option_postprocess

CB_reader_cfg = dict(
    input_columns=['premise', 'hypothesis'],
    output_column='label',
)

CB_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=
                    '{premise}\n{hypothesis}\nWhat is the relation between the two sentences?\nA. Contradiction\nB. Entailment\nC. Neutral\nAnswer:'
                ),
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

CB_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABC'),
)

CB_datasets = [
    dict(
        abbr='CB',
        type=CBDatasetV2,
        path='./data/SuperGLUE/CB/val.jsonl',
        reader_cfg=CB_reader_cfg,
        infer_cfg=CB_infer_cfg,
        eval_cfg=CB_eval_cfg,
    )
]
