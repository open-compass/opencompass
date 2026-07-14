from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import UMG_Dataset, UMG_Evaluator

INFER_TEMPLATE = '''Generate a molecule with <SMILES>'''

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
        type=UMG_Evaluator,
    ),
)

UMG_Datasets = [
    dict(
        abbr='SciReasoner-unconditional_molecule_generation',
        type=UMG_Dataset,
        # max_cut=20,  # Optionally limit the maximum number of samples
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg)
]
mini_UMG_Datasets = [
    dict(
        abbr='SciReasoner-unconditional_molecule_generation-mini',
        type=UMG_Dataset,
        max_cut=150,  # Optionally limit the maximum number of samples
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg)
]
