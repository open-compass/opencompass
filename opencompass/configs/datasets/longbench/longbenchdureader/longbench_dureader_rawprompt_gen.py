from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchRougeEvaluator, LongBenchdureaderDataset

LongBench_dureader_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LongBench_dureader_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBench_dureader_eval_cfg = dict(
    evaluator=dict(type=LongBenchRougeEvaluator, language='zh'), pred_role='BOT'
)

LongBench_dureader_datasets = [
    dict(
        type=LongBenchdureaderDataset,
        abbr='LongBench_dureader',
        path='opencompass/Longbench',
        name='dureader',
        reader_cfg=LongBench_dureader_reader_cfg,
        infer_cfg=LongBench_dureader_infer_cfg,
        eval_cfg=LongBench_dureader_eval_cfg,
    )
]
