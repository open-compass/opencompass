from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRougeEvaluator, LongBenchmulti_newsDataset

LongBench_multi_news_reader_cfg = dict(
    input_columns=['context'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LongBench_multi_news_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': 'You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:\n'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBench_multi_news_eval_cfg = dict(
    evaluator=dict(type=LongBenchRougeEvaluator), pred_role='BOT'
)

LongBench_multi_news_datasets = [
    dict(
        type=LongBenchmulti_newsDataset,
        abbr='LongBench_multi_news',
        path='opencompass/Longbench',
        name='multi_news',
        reader_cfg=LongBench_multi_news_reader_cfg,
        infer_cfg=LongBench_multi_news_infer_cfg,
        eval_cfg=LongBench_multi_news_eval_cfg,
    )
]
