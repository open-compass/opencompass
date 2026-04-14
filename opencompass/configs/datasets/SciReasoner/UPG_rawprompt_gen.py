from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import UPGDataset, UPG_postprocess, UPG_Evaluator

reader_cfg = dict(input_columns=['input'], output_column='output')

infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{input}'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
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