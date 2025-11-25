from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.datasets import Uncond_material_Dataset, uncond_material_Evaluator, material_postprocessor

uncond_material_reader_cfg = dict(input_columns=['input'], output_column='output')

generation_kwargs = dict(
    do_sample=True,
    top_p=1,
    temperature=1.8,
    top_k=80,
)

uncond_material_infer_cfg = dict(
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
    inferencer=dict(type=GenInferencer, generation_kwargs=generation_kwargs),
)

uncond_material_eval_cfg = dict(
            evaluator=dict(type=uncond_material_Evaluator),
            pred_postprocessor=dict(type=material_postprocessor),
        )

uncond_material_datasets = [
    dict(
        abbr='unconditional_material_generation',
        type=Uncond_material_Dataset,
        num=5000,
        prompt='Produce a material that has any bulk modulus or composition',
        reader_cfg=uncond_material_reader_cfg,
        infer_cfg=uncond_material_infer_cfg,
        eval_cfg=uncond_material_eval_cfg,
    )
]
mini_uncond_material_datasets = [
    dict(
        abbr='unconditional_material_generation-mini',
        type=Uncond_material_Dataset,
        num=150,
        prompt='Produce a material that has any bulk modulus or composition',
        reader_cfg=uncond_material_reader_cfg,
        infer_cfg=uncond_material_infer_cfg,
        eval_cfg=uncond_material_eval_cfg,
    )
]
