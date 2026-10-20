from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.s3eval import S3EvalDataset, S3EvalEvaluator


s3eval_cfg = dict(evaluator=dict(type=S3EvalEvaluator))


s3eval_datasets = [
    dict(
        type=S3EvalDataset,
        abbr='s3eval',
        path='FangyuLei/s3eval',
        eval_cfg=s3eval_cfg)
]
