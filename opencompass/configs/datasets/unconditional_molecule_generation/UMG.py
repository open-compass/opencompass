from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import UMG_Dataset, UMG_Evaluator

INFER_TEMPLATE = '''Generate a molecule with <SMILES>'''

reader_cfg = dict(input_columns=['input'], output_column='output')

generation_kwargs = dict(
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.95,
    # eos_token_id=[151643, 151645], 
)

infer_cfg = dict(
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

eval_cfg = dict(
    evaluator=dict(
        type=UMG_Evaluator,
    ),
)

UMG_Datasets = [
    dict(
        abbr='unconditional_molecule_generation',
        type=UMG_Dataset,
        # max_cut=20,  # Optionally limit the maximum number of samples
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg)
]
mini_UMG_Datasets = [
    dict(
        abbr='unconditional_molecule_generation-mini',
        type=UMG_Dataset,
        max_cut=150,  # Optionally limit the maximum number of samples
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg)
]
