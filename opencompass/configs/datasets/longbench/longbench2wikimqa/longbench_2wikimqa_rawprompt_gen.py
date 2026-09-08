from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchF1Evaluator, LongBench2wikimqaDataset

LongBench_2wikimqa_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LongBench_2wikimqa_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBench_2wikimqa_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator), pred_role='BOT'
)

LongBench_2wikimqa_datasets = [
    dict(
        type=LongBench2wikimqaDataset,
        abbr='LongBench_2wikimqa',
        path='opencompass/Longbench',
        name='2wikimqa',
        reader_cfg=LongBench_2wikimqa_reader_cfg,
        infer_cfg=LongBench_2wikimqa_infer_cfg,
        eval_cfg=LongBench_2wikimqa_eval_cfg,
    )
]
