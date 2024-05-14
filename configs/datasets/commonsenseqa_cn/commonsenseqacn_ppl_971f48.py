from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CommonsenseQADataset_CN

commonsenseqacn_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D', 'E'],
    output_column='answerKey',
    test_split='validation',
)

_ice_template = dict(
    type=PromptTemplate,
    template={
        ans: dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt='问题: {question}\n答案: '),
                dict(role='BOT', prompt=ans_token),
            ],
        )
        for ans, ans_token in [
            ['A', '{A}'],
            ['B', '{B}'],
            ['C', '{C}'],
            ['D', '{D}'],
            ['E', '{E}'],
        ]
    },
    ice_token='</E>',
)


commonsenseqacn_infer_cfg = dict(
    prompt_template=_ice_template,
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

commonsenseqacn_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

commonsenseqacn_datasets = [
    dict(
        abbr='commonsenseqa_cn',
        type=CommonsenseQADataset_CN,
        path='./data/commonsenseqa_cn/validation.jsonl',
        reader_cfg=commonsenseqacn_reader_cfg,
        infer_cfg=commonsenseqacn_infer_cfg,
        eval_cfg=commonsenseqacn_eval_cfg,
    )
]
