from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

from opencompass.datasets.sage.prompt import SAGE_INFER_TEMPLATE
from opencompass.datasets.sage.dataset_loader import SAGEDataset


compass_agi4s_reader_cfg = dict(
    input_columns=['problem'], 
    output_column='answer'
)

compass_agi4s_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=SAGE_INFER_TEMPLATE,
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

sage_datasets = [
    dict(
        type=SAGEDataset,
        n=4,
        abbr='sage-test',
        split='test',
        reader_cfg=compass_agi4s_reader_cfg,
        infer_cfg=compass_agi4s_infer_cfg,
        eval_cfg=dict(),
    )
]

datasets = sage_datasets