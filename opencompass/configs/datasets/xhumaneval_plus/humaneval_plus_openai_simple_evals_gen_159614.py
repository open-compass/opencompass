# THIS SHALL ALSO BE DEPRECATED
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HumanevalDataset, HumanEvalPlusEvaluator, humaneval_postprocess_v2

humaneval_plus_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test')

# TODO: allow empty output-column
humaneval_plus_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n{prompt}'
            ),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

humaneval_plus_eval_cfg = dict(
    evaluator=dict(type=HumanEvalPlusEvaluator),
    pred_role='BOT',
    k=[1, 10, 100],  # the parameter only for humaneval
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)

humaneval_plus_datasets = [
    dict(
        abbr='humaneval_plus',
        type=HumanevalDataset,
        path='opencompass/humaneval',
        reader_cfg=humaneval_plus_reader_cfg,
        infer_cfg=humaneval_plus_infer_cfg,
        eval_cfg=humaneval_plus_eval_cfg)
]
