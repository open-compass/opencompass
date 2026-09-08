from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchCodeSimEvaluator, LongBenchrepobenchDataset

LongBench_repobench_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LongBench_repobench_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': 'Please complete the code given below. \n{context}{input}Next line of code:\n'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBench_repobench_eval_cfg = dict(
    evaluator=dict(type=LongBenchCodeSimEvaluator), pred_role='BOT'
)

LongBench_repobench_datasets = [
    dict(
        type=LongBenchrepobenchDataset,
        abbr='LongBench_repobench-p',
        path='opencompass/Longbench',
        name='repobench-p',
        reader_cfg=LongBench_repobench_reader_cfg,
        infer_cfg=LongBench_repobench_infer_cfg,
        eval_cfg=LongBench_repobench_eval_cfg,
    )
]
