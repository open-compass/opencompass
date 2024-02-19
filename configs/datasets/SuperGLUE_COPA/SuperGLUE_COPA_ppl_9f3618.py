from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

COPA_reader_cfg = dict(
    input_columns=['question', 'premise', 'choice1', 'choice2'],
    output_column='label',
    test_split='train')

COPA_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0:
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt='{premise}\nQuestion: What may be the {question}?\nAnswer:'),
                dict(role='BOT', prompt='{choice1}'),
            ]),
            1:
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt='{premise}\nQuestion: What may be the {question}?\nAnswer:'),
                dict(role='BOT', prompt='{choice2}'),
            ]),
        },
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

COPA_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

COPA_datasets = [
    dict(
        type=HFDataset,
        abbr='COPA',
        path='json',
        data_files='./data/SuperGLUE/COPA/val.jsonl',
        split='train',
        reader_cfg=COPA_reader_cfg,
        infer_cfg=COPA_infer_cfg,
        eval_cfg=COPA_eval_cfg,
    )
]
