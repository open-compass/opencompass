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
        use_extract_model=False,
        extract_url=[],
        extract_model_name='',
        k=[4, 8, 16],
        replication=3,
        thresholds=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
)

livemathbench_datasets = [
    dict(
        type=LiveMathBenchDataset,
        path='',
        k=16,
        replication=3,
        dataset_splits=['hard'],
        dataset_languages=['cn', 'en'],
        cot=True,
        version='202412',
        abbr='LiveMathBench-v202412-hard-k16r3',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg
    )
]