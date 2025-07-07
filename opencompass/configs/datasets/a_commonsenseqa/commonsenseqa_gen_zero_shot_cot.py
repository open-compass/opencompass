# Use FixKRetriever to avoid hang caused by the Huggingface
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import commonsenseqaDataset
from opencompass.utils.text_postprocessors import (
    match_answer_pattern,
)

commonsenseqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D', 'E'],
    output_column='answerKey',
    test_split='validation')

commonsenseqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nPlease reason step by step, and finally answer with the correct letter choice in the following format:\nSo the answer is {your chosen option}.'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

commonsenseqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(
        type=match_answer_pattern, answer_pattern=r'(?i)so the answer is\s*([A-P])'
    ),
)

commonsenseqa_datasets = [
    dict(
        abbr='commonsense_qa_zero_shot_cot',
        type=commonsenseqaDataset,
        path='opencompass/commonsense_qa',
        reader_cfg=commonsenseqa_reader_cfg,
        infer_cfg=commonsenseqa_infer_cfg,
        eval_cfg=commonsenseqa_eval_cfg,
    )
]

