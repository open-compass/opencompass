from mmengine.config import read_base
with read_base():
    from ..utils.utils import get_data_path
# THIS SHALL ALSO BE DEPRECATED
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalDataset, HumanEvalEvaluator, humaneval_postprocess_v2

humaneval_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test')

# TODO: allow empty output-column
humaneval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='Complete the following python code:\n{prompt}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

humaneval_eval_cfg = dict(
    evaluator=dict(type=HumanEvalEvaluator),
    pred_role='BOT',
    k=[1, 10, 100],  # the parameter only for humaneval
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)

humaneval_datasets = [
    dict(
        abbr='openai_humaneval',
        type=HumanevalDataset,
        path=get_data_path('opencompass/humaneval', './data/humaneval/human-eval-v2-20210705.jsonl'),
        reader_cfg=humaneval_reader_cfg,
        infer_cfg=humaneval_infer_cfg,
        eval_cfg=humaneval_eval_cfg)
]
