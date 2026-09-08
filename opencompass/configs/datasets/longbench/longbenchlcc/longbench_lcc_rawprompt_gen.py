from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchCodeSimEvaluator, LongBenchlccDataset

LongBench_lcc_reader_cfg = dict(
    input_columns=['context'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LongBench_lcc_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': 'Please complete the code given below. \n{context}Next line of code:\n'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBench_lcc_eval_cfg = dict(
    evaluator=dict(type=LongBenchCodeSimEvaluator), pred_role='BOT'
)

LongBench_lcc_datasets = [
    dict(
        type=LongBenchlccDataset,
        abbr='LongBench_lcc',
        path='opencompass/Longbench',
        name='lcc',
        reader_cfg=LongBench_lcc_reader_cfg,
        infer_cfg=LongBench_lcc_infer_cfg,
        eval_cfg=LongBench_lcc_eval_cfg,
    )
]
