from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import Infer_COREV2_Dataset

corev2_reader_cfg = dict(
    input_columns=["question"],
    output_column='judge'
    )

corev2_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt="{question}"
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


infer_corev2_datasets = [
    dict(
        type=Infer_COREV2_Dataset,
        path="./data/subject/corev2",
        reader_cfg=corev2_reader_cfg,
        infer_cfg=corev2_infer_cfg,
        )
]

