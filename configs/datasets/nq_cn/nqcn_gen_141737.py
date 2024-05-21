from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import NaturalQuestionDataset_CN, NQEvaluator_CN

nqcn_reader_cfg = dict(
    input_columns=['question'], output_column='answer', train_split='test'
)

nqcn_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='问题: {question}?\n答案是：'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

nqcn_eval_cfg = dict(evaluator=dict(type=NQEvaluator_CN), pred_role='BOT')

nqcn_datasets = [
    dict(
        abbr='nq_cn',
        type=NaturalQuestionDataset_CN,
        path='./data/nq_cn',
        reader_cfg=nqcn_reader_cfg,
        infer_cfg=nqcn_infer_cfg,
        eval_cfg=nqcn_eval_cfg,
    )
]
