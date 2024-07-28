from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CMNLIDatasetV2
from opencompass.utils.text_postprocessors import first_capital_postprocess

cmnli_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',
    test_split='train')

cmnli_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                '阅读文章：{sentence1}\n根据上文，回答如下问题：{sentence2}\nA. 对\nB. 错\nC. 可能\n请从“A”，“B”，“C”中进行选择。\n答：'
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

cmnli_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

cmnli_datasets = [
    dict(
        abbr='cmnli',
        type=CMNLIDatasetV2,
        path='opencompass/cmnli-dev',
        reader_cfg=cmnli_reader_cfg,
        infer_cfg=cmnli_infer_cfg,
        eval_cfg=cmnli_eval_cfg,
    )
]
