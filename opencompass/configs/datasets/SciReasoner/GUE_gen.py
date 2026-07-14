from opencompass.datasets import (
    GUE_Dataset,
    GUE_Evaluator,
    GUE_postprocessor
)
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever

GUE_sub_tasks = [
    'cpd-prom_core_all',
    'cpd-prom_core_notata',
    'cpd-prom_core_tata',
    'pd-prom_300_all',
    'pd-prom_300_notata',
    'pd-prom_300_tata',
    'tf-h-0',
    'tf-h-1',
    'tf-h-2',
    'tf-h-3',
    'tf-h-4',
]

GUE_reader_cfg = dict(input_columns=['input'], output_column='output')

GUE_datasets = []
mini_GUE_datasets = []

for name in GUE_sub_tasks:

    GUE_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='{input}'),
                ]
            ),
        ),
        retriever=dict(
            type=ZeroRetriever
        ),
        inferencer=dict(type=GenInferencer),
    )

    GUE_eval_cfg = dict(
        evaluator=dict(
            type=GUE_Evaluator
        ),
        pred_role='BOT',
        pred_postprocessor=dict(type=GUE_postprocessor),
        dataset_postprocessor=dict(type=GUE_postprocessor),
    )

    GUE_datasets.append(
        dict(
            abbr=f'SciReasoner-Gue_{name}',
            type=GUE_Dataset,
            path='opencompass/SciReasoner-GUE',
            task=name,
            reader_cfg=GUE_reader_cfg,
            infer_cfg=GUE_infer_cfg,
            eval_cfg=GUE_eval_cfg,
        )
    )
    mini_GUE_datasets.append(
        dict(
            abbr=f'SciReasoner-Gue_{name}-mini',
            type=GUE_Dataset,
            path='opencompass/SciReasoner-GUE',
            task=name,
            mini_set=True,
            reader_cfg=GUE_reader_cfg,
            infer_cfg=GUE_infer_cfg,
            eval_cfg=GUE_eval_cfg,
        )
    )
