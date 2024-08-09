from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import IFEvalDataset, IFEvaluator

ifeval_reader_cfg = dict(
    input_columns=['prompt'], output_column='reference')

ifeval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='{prompt}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1025))

ifeval_eval_cfg = dict(
    evaluator=dict(type=IFEvaluator),
    pred_role='BOT',
)

ifeval_datasets = [
    dict(
        abbr='IFEval',
        type=IFEvalDataset,
        path='data/ifeval/input_data.jsonl',
        reader_cfg=ifeval_reader_cfg,
        infer_cfg=ifeval_infer_cfg,
        eval_cfg=ifeval_eval_cfg)
]
