from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CommonsenseQADataset_CN
from opencompass.utils.text_postprocessors import first_capital_postprocess

commonsenseqacn_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D', 'E'],
    output_column='answerKey',
    test_split='validation',
)

_ice_template = dict(
    type=PromptTemplate,
    template=dict(
        begin='</E>',
        round=[
            dict(
                role='HUMAN',
                prompt='{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\n答案：',
            ),
            dict(role='BOT', prompt='{answerKey}'),
        ],
    ),
    ice_token='</E>',
)


commonsenseqacn_infer_cfg = dict(
    prompt_template=_ice_template,
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

commonsenseqacn_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_capital_postprocess),
)

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
