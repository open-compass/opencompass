from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WiCDatasetV2
from opencompass.utils.text_postprocessors import first_capital_postprocess

WiC_reader_cfg = dict(
    input_columns=[
        'word',
        'sentence1',
        'sentence2',
    ],
    output_column='label',
)

WiC_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nAre '{word}' in the above two sentenses the same?\nA. Yes\nB. No\nAnswer:"
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

WiC_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

WiC_datasets = [
    dict(
        abbr='WiC',
        type=WiCDatasetV2,
        path='./data/SuperGLUE/WiC/val.jsonl',
        reader_cfg=WiC_reader_cfg,
        infer_cfg=WiC_infer_cfg,
        eval_cfg=WiC_eval_cfg,
    )
]
