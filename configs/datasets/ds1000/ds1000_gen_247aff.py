from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (DS1000Dataset_Interperter,
                                  DS1000InterpreterEvaluator)

ds1000_reader_cfg = dict(
    input_columns=["prompt"],
    output_column="test_column",
    train_split='test',
    test_split='test')

ds1000_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{prompt}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

ds1000_eval_cfg = dict(
    evaluator=dict(type=DS1000InterpreterEvaluator),
    pred_role="BOT",
)

# Matplotlib cannot fit this setting
ds1000_datasets = [
    dict(
        abbr=f"ds1000_{lib}",
        type=DS1000Dataset_Interperter,  # bustm share the same format with AFQMC
        path="ds1000_data/",
        libs=f"{lib}",
        reader_cfg=ds1000_reader_cfg,
        infer_cfg=ds1000_infer_cfg,
        eval_cfg=ds1000_eval_cfg,
    ) for lib in [
        'Pandas',
        'Numpy',
        'Tensorflow',
        'Scipy',
        'Sklearn',
        'Pytorch',
    ]
]
