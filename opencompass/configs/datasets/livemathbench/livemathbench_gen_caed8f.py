from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

from opencompass.datasets.livemathbench import LiveMathBenchDataset, LiveMathBenchEvaluator


livemathbench_reader_cfg = dict(
    input_columns=['prompt'], 
    output_column='answer'
)

livemathbench_infer_cfg = dict(
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
        max_out_len=2048,
        temperature=1.0
    )
)

livemathbench_eval_cfg = dict(
    evaluator=dict(
        type=LiveMathBenchEvaluator, 
        model_name='Qwen/Qwen2.5-72B-Instruct', 
        url=[]
    )
)

livemathbench_datasets = [
    dict(
        type=LiveMathBenchDataset,
        abbr='LiveMathBench',
        path='',
        k=32,
        n=5,
        reader_cfg=livemathbench_reader_cfg,
        infer_cfg=livemathbench_infer_cfg,
        eval_cfg=livemathbench_eval_cfg
    )
]