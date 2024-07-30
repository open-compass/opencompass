from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets import Py150Dataset

from opencompass.utils.text_postprocessors import first_capital_postprocess


py150_reader_cfg = dict(
    input_columns='input',
    output_column='gt',
)

py150_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="I will give you a part of python code. Please write down what the next line of code is. Note that you only need to give the next line of code, and you don't need to give any other reply.\nCode:{input}\nNext line:"),
                dict(role='BOT', prompt='{gt}'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

py150_eval_cfg = dict(evaluator=dict(type=BleuEvaluator),
                        pred_role='BOT'
                    )

py150_datasets = [
    dict(
        type=Py150Dataset,
        abbr=f'py150',
        path=f'data/py150/test.json',
        reader_cfg=py150_reader_cfg,
        infer_cfg=py150_infer_cfg,
        eval_cfg=py150_eval_cfg,
    )
]
