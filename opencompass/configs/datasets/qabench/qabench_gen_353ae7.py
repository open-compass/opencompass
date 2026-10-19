from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HFDataset

qabench_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='reference',
)

# TODO: allow empty output-column
qabench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='{prompt}')])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

qabench_datasets = [
    dict(
        type=HFDataset,
        path='csv',
        data_files='./data/qabench/qabench-test.qa.csv',
        abbr='qabench',
        split='train',
        reader_cfg=qabench_reader_cfg,
        infer_cfg=qabench_infer_cfg,
        eval_cfg=dict(ds_column='reference'))
]
