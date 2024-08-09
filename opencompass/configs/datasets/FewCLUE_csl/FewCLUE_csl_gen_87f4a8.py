from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CslDatasetV2
from opencompass.utils.text_postprocessors import first_capital_postprocess

csl_reader_cfg = dict(
    input_columns=['abst', 'keywords'],
    output_column='label',
)

csl_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                '摘要：{abst}\n关键词：{keywords}\n上述关键词出现在学术期刊中是否恰当？\nA. 否\nB. 是\n请从”A“，”B“中进行选择。\n答：'
            )
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

csl_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

csl_datasets = [
    dict(
        abbr='csl_dev',
        type=CslDatasetV2,
        path='./data/FewCLUE/csl/dev_few_all.json',
        reader_cfg=csl_reader_cfg,
        infer_cfg=csl_infer_cfg,
        eval_cfg=csl_eval_cfg,
    ),
    dict(
        abbr='csl_test',
        type=CslDatasetV2,
        path='./data/FewCLUE/csl/test_public.json',
        reader_cfg=csl_reader_cfg,
        infer_cfg=csl_infer_cfg,
        eval_cfg=csl_eval_cfg,
    ),
]
