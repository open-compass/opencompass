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
        max_out_len=16384,
        temperature=1.0
    )
)

livemathbench_eval_cfg = dict(
    evaluator=dict(
        type=LiveMathBenchEvaluator, 
        model_name='Qwen/Qwen2.5-72B-Instruct', 
        url=['http://172.30.40.154:23333/v1/'] #'https://api.openai.com/v1/'
    )
)

livemathbench_datasets = [
    dict(
        type=LiveMathBenchDataset,
        abbr='LiveMathBench-k1-n1',
        path='opencompass/LiveMathBench202412',
        k=1, # K@Pass
        n=1,  # Run times
        reader_cfg=livemathbench_reader_cfg,
        infer_cfg=livemathbench_infer_cfg,
        eval_cfg=livemathbench_eval_cfg
    )
]