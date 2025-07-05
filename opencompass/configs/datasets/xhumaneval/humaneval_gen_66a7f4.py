from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalDataset, HumanEvalEvaluator, humaneval_postprocess_v2

humaneval_reader_cfg = dict(input_columns=['prompt'], output_column='task_id', train_split='test')

HUMANEVAL_TEMPLATE = dict(
    round=[
        dict(role='HUMAN', prompt='You are an intelligent programming assistant to produce Python algorithmic solutions.\nCan you complete the following Python function?\n```python\n{prompt}\n```'),
    ]
)

humaneval_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=HUMANEVAL_TEMPLATE),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

humaneval_eval_cfg = dict(
    evaluator=dict(type=HumanEvalEvaluator),
    k=[1, 10, 100],
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)

humaneval_datasets = [
    dict(
        abbr='openai_humaneval',
        type=HumanevalDataset,
        path='opencompass/humaneval',
        reader_cfg=humaneval_reader_cfg,
        infer_cfg=humaneval_infer_cfg,
        eval_cfg=humaneval_eval_cfg,
    )
]
