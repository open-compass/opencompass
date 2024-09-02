from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TACODataset, TACOEvaluator

TACO_difficulties_list = ['EASY', 'MEDIUM', 'MEDIUM_HARD', 'HARD', 'VERY_HARD']
TACO_reader_cfg = dict(input_columns=['question', 'starter'], output_column='problem_id', train_split='test')

TACO_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Please write a python program to address the following QUESTION. Your ANSWER should be in a code block format like this: ```python # Write your code here ```. \nQUESTION:\n{question} {starter}\nANSWER:\n'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

TACO_eval_cfg = dict(evaluator=dict(type=TACOEvaluator), pred_role='BOT')

TACO_datasets = []
for difficulty in TACO_difficulties_list:
    TACO_datasets.append(
        dict(
            type=TACODataset,
            abbr='TACO-' + difficulty,
            path='./data/BAAI-TACO',
            difficulty=difficulty,
            reader_cfg=TACO_reader_cfg,
            infer_cfg=TACO_infer_cfg,
            eval_cfg=TACO_eval_cfg,
        )
    )
