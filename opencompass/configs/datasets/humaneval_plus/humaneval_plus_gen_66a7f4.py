from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalDataset, HumanEvalPlusEvaluator, humaneval_postprocess_v2

humaneval_plus_reader_cfg = dict(input_columns=['prompt'], output_column='task_id', train_split='test')

HUMANEVAL_TEMPLATE = dict(
    round=[
        dict(role='HUMAN', prompt='You are an intelligent programming assistant to produce Python algorithmic solutions.\nCan you complete the following Python function?\n```python\n{prompt}\n```'),
    ]
)

humaneval_plus_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=HUMANEVAL_TEMPLATE),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

humaneval_plus_eval_cfg = dict(
    evaluator=dict(type=HumanEvalPlusEvaluator),
    k=[1, 10, 100],
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)

humaneval_plus_datasets = [
    dict(
        abbr='humaneval_plus',
        type=HumanevalDataset,
        path='opencompass/humaneval',
        reader_cfg=humaneval_plus_reader_cfg,
        infer_cfg=humaneval_plus_infer_cfg,
        eval_cfg=humaneval_plus_eval_cfg,
    )
]
