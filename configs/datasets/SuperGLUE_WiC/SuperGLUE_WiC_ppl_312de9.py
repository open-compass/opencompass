from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WiCDataset

WiC_reader_cfg = dict(
    input_columns=[
        'word',
        'sentence1',
        'sentence2',
    ],
    output_column='answer',
    test_split='train')

WiC_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0:
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    "Sentence 1: {sentence1}\nSentence 2: {sentence2}\n'{word}' in the above two sentenses are different."
                ),
            ]),
            1:
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    "Sentence 1: {sentence1}\nSentence 2: {sentence2}\n'{word}' in the above two sentenses are the same."
                ),
            ]),
        },
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

WiC_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

WiC_datasets = [
    dict(
        type=WiCDataset,
        abbr='WiC',
        path='json',
        data_files='./data/SuperGLUE/WiC/val.jsonl',
        split='train',
        reader_cfg=WiC_reader_cfg,
        infer_cfg=WiC_infer_cfg,
        eval_cfg=WiC_eval_cfg,
    )
]
