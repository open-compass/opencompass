from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import UPGDataset, UPG_postprocess, UPG_Evaluator

reader_cfg = dict(input_columns=['input'], output_column='output')

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                '</E>',
            ],
            round=[
                dict(role='HUMAN', prompt='{input}'),
            ]
        ),
        ice_token='</E>',
    ),
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                # dict(role='HUMAN', prompt='{input} /no_think'), # for Qwen3
                dict(role='HUMAN', prompt='{input}'),
                dict(role='BOT', prompt='{output}'),
            ]
        )
    ),
    # The retriever is responsible for retrieving examples and formatting them using ice_template
    retriever=dict(
        # type=FixKRetriever,
        # fix_id_list=[0, 1, 2, 3, 4], # Use the first 5 examples
        type=ZeroRetriever,  # For our trained models, use zero-shot
    ),
    inferencer=dict(
        type=GenInferencer,
    ),
)

eval_cfg = dict(
    evaluator=dict(
        type=UPG_Evaluator,
    ),
    pred_postprocessor=dict(type=UPG_postprocess),
    dataset_postprocessor=dict(type=UPG_postprocess),
)

UPG_datasets = [
    dict(
        abbr='SciReasoner-unconditional_protein_generation',
        type=UPGDataset,
        # max_cut=20,  # Optionally limit the maximum number of samples
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg)
]
mini_UPG_datasets = [
    dict(
        abbr='SciReasoner-unconditional_protein_generation-mini',
        type=UPGDataset,
        max_cut=150,  # Optionally limit the maximum number of samples
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg)
]
