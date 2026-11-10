from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    LongBenchClassificationEvaluator,
    LongBenchtrecDataset,
    trec_postprocess,
)

LongBench_trec_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='all_labels',
    train_split='test',
    test_split='test',
)

LongBench_trec_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': 'Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBench_trec_eval_cfg = dict(
    evaluator=dict(type=LongBenchClassificationEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=trec_postprocess),
)

LongBench_trec_datasets = [
    dict(
        type=LongBenchtrecDataset,
        abbr='LongBench_trec',
        path='opencompass/Longbench',
        name='trec',
        reader_cfg=LongBench_trec_reader_cfg,
        infer_cfg=LongBench_trec_infer_cfg,
        eval_cfg=LongBench_trec_eval_cfg,
    )
]
