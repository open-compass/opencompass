from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

from opencompass.datasets.livemathbench import LiveMathBenchDataset, LiveMathBenchEvaluator

reader_cfg = dict(
    input_columns=['prompt'], 
    output_column='answer'
)

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer, 
        max_out_len=8192
    ),
)

eval_cfg = dict(
    evaluator=dict(
        type=LiveMathBenchEvaluator,
        model_name='',
        url=[],
        k=[1],
        replication=1,
        thresholds=[1.0]
    )
)

livemathbench_datasets = [
    dict(
        type=LiveMathBenchDataset,
        path='',
        k=1,
        replication=1,
        dataset_splits=['CNMO', 'CCEE', 'AMC', 'WLPMC'],
        dataset_languages=['cn', 'en'],
        cot=True,
        version='202412',
        abbr='LiveMathBench-v202412-k1r1',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg
    )
]