from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

from opencompass.datasets.livemathbench import LiveMathBenchDataset, LiveMathBenchEvaluator


livemathbench_dataset = dict(
    type=LiveMathBenchDataset,
    path='',
    k=1,
    replication=1,
    dataset_splits=['CNMO', 'CCEE', 'AMC', 'WLPMC'],
    dataset_languages=['cn', 'en'],
    cot=True,
    version='202412',
    abbr='LiveMathBench-v202412',
    reader_cfg=dict(
        input_columns=['prompt'], 
        output_column='answer'
    ),
    infer_cfg=dict(
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
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=LiveMathBenchEvaluator,
            model_name='',
            url=[],
            use_extract_model=False,
            extract_url=[],
            extract_model_name='',
            k=[1],
            replication=1,
            thresholds=[0.0]
        )
    )
)
livemathbench_datasets = [livemathbench_dataset]