from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.datasets import Bulk_modulus_material_Dataset, material_Evaluator, material_postprocessor

modulus_material_reader = dict(input_columns=['input'], output_column='output')

modulus_material_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{input}',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

modulus_material_eval_cfg = dict(
    evaluator=dict(
        type=material_Evaluator,
        data_path='opencompass/SciReasoner-Conditional_generation',
    ),
    pred_postprocessor=dict(type=material_postprocessor),
)

modulus_material_datasets = [
    dict(
        abbr='SciReasoner-bulk_modulus_to_material_generation',
        type=Bulk_modulus_material_Dataset,
        path='opencompass/SciReasoner-Conditional_generation',
        reader_cfg=modulus_material_reader,
        infer_cfg=modulus_material_infer_cfg,
        eval_cfg=modulus_material_eval_cfg,
    )
]
mini_modulus_material_datasets = [
    dict(
        abbr='SciReasoner-bulk_modulus_to_material_generation-mini',
        type=Bulk_modulus_material_Dataset,
        path='opencompass/SciReasoner-Conditional_generation',
        mini_set=True,
        reader_cfg=modulus_material_reader,
        infer_cfg=modulus_material_infer_cfg,
        eval_cfg=modulus_material_eval_cfg,
    )
]
