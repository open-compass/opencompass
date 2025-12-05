from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    OJBenchDataset,
)

ojbench_reader_cfg = dict(
    input_columns=[
        'prompt',
    ],
    output_column='id',
)

ojbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}'
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

ojbench_eval_cfg = dict(
    evaluator=dict()
)

ojbench_datasets = [
    dict(
        type=OJBenchDataset,
        abbr='ojbench',
        path='opencompass/ojbench',
        reader_cfg=ojbench_reader_cfg,
        infer_cfg=ojbench_infer_cfg,
        eval_cfg=ojbench_eval_cfg,
    )
]