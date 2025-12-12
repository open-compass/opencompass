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

generation_kwargs = dict(
    do_sample=False,
)

for name in GUE_sub_tasks:
    train_path = f'/path/GUE-test/{name}/dev/data.json'
    test_path = f'/path/GUE-test/{name}/test/data.json'

    GUE_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='{input}'),
                    dict(role='BOT', prompt=''),
                ]
            ),
        ),
        retriever=dict(
            type=ZeroRetriever
        ),
        inferencer=dict(type=GenInferencer, generation_kwargs=generation_kwargs),
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
            abbr=name,
            type=GUE_Dataset,
            train_path=train_path,
            test_path=test_path,
            hf_hub=False,
            reader_cfg=GUE_reader_cfg,
            infer_cfg=GUE_infer_cfg,
            eval_cfg=GUE_eval_cfg,
        )
    )
    mini_GUE_datasets.append(
        dict(
            abbr=f'{name}-mini',
            type=GUE_Dataset,
            train_path=train_path,
            test_path=test_path,
            mini_set=True,
            hf_hub=False,
            reader_cfg=GUE_reader_cfg,
            infer_cfg=GUE_infer_cfg,
            eval_cfg=GUE_eval_cfg,
        )
    )
