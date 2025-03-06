from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

from opencompass.datasets.livemathbench import LiveMathBenchDataset, LiveMathBenchEvaluator


livemathbench_dataset = dict(
    type=LiveMathBenchDataset,
    path='',
    k=1,
    n=1,
    dataset_splits=['hard'],
    dataset_languages=['cn', 'en'],
    cot=True,
    version='202412',
    abbr='LiveMathBench-v202412-Hard',
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
            type=GenInferencer
        ),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=LiveMathBenchEvaluator,
            model_name='',
            url=[]
        )
    )
)
livemathbench_datasets = [livemathbench_dataset]