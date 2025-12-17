from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.datasets import Composition_material_Dataset, composition_Evaluator, material_postprocessor

generation_kwargs = dict(
    do_sample=True,
    # top_p=0.8,
    # min_p=0,
    temperature=0.40,
    # top_k=20,
    # repetition_penalty=1,
    # "<|endoftext|>": 151643 "<|im_end|>": 151645
    # eos_token_id=[151643, 151645],
)

composition_material_reader = dict(input_columns=['input'], output_column='output')

composition_material_infer_cfg = dict(
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

composition_material_eval_cfg = dict(
    evaluator=dict(
        type=composition_Evaluator,
        data_path='opencompass/SciReasoner-Conditional_generation',
    ),
    pred_postprocessor=dict(type=material_postprocessor),
)


composition_material_datasets = [
    dict(
        abbr='SciReasoner-composition_to_material_generation',
        type=Composition_material_Dataset,
        path='opencompass/SciReasoner-Conditional_generation',
        reader_cfg=composition_material_reader,
        infer_cfg=composition_material_infer_cfg,
        eval_cfg=composition_material_eval_cfg,
    )
]
mini_composition_material_datasets = [
    dict(
        abbr='SciReasoner-composition_to_material_generation-mini',
        type=Composition_material_Dataset,
        path='opencompass/SciReasoner-Conditional_generation',
        mini_set=True,
        reader_cfg=composition_material_reader,
        infer_cfg=composition_material_infer_cfg,
        eval_cfg=composition_material_eval_cfg,
    )
]
