from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SubInfer_Dataset

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
        type=SubInfer_Dataset,
        path="./data/subject/corev2/COREV2_6A.json",
        reader_cfg=corev2_reader_cfg,
        infer_cfg=corev2_infer_cfg,
        )
]

