from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import LastLettersDataset, last_letters_pred_postprocess

last_letters_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='test',
    test_split='test'
)

last_letters_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Question: {question}\nPlease reason step by step, and format your final answer as `The answer is [ANSWER]`.\nAnswer:'),
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

last_letters_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=last_letters_pred_postprocess),
)

last_letters_datasets = [
    dict(
        abbr='last_letters',
        type=LastLettersDataset,
        path='last_letters',
        reader_cfg=last_letters_reader_cfg,
        infer_cfg=last_letters_infer_cfg,
        eval_cfg=last_letters_eval_cfg
    )
]