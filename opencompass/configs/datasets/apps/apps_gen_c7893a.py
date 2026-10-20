from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import APPSDataset, APPSEvaluator

APPS_reader_cfg = dict(input_columns=['question', 'starter'], output_column='problem_id', train_split='test')

APPS_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='Please write a python program to address the following QUESTION. Your ANSWER should be in a code block format like this: ```python # Write your code here ```. \nQUESTION:\n{question} {starter}\nANSWER:\n'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

APPS_eval_cfg = dict(evaluator=dict(type=APPSEvaluator), pred_role='BOT')

APPS_datasets = [
    dict(
        type=APPSDataset,
        abbr='apps',
        path='codeparrot/apps',
        num_repeats=1,
        reader_cfg=APPS_reader_cfg,
        infer_cfg=APPS_infer_cfg,
        eval_cfg=APPS_eval_cfg,
    )
]
