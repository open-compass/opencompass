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
                '摘要是对论文内容不加注释和评论的简短陈述，要求扼要地说明研究工作的目的、研究方法和最终结论等。\n关键词是一篇学术论文的核心词汇，一般由一系列名词组成。关键词在全文中应有较高出现频率，且能起到帮助文献检索的作用。\n摘要：{abst}\n关键词：{keywords}\n请问上述关键词是否匹配摘要且符合要求？\nA. 否\nB. 是\n请从”A“，”B“中进行选择。\n答：'
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
