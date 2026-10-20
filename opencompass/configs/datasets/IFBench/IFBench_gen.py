from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import IFEvalDataset, IFBenchEvaluator

ifbench_reader_cfg = dict(
    input_columns=['prompt'], output_column='reference')

ifbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='{prompt}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

ifbench_eval_cfg = dict(
    evaluator=dict(type=IFBenchEvaluator),
    pred_role='BOT',
)

ifbench_datasets = [
    dict(
        abbr='IFBench',
        type=IFEvalDataset,
        path='opencompass/IFbench',
        reader_cfg=ifbench_reader_cfg,
        infer_cfg=ifbench_infer_cfg,
        eval_cfg=ifbench_eval_cfg)
]