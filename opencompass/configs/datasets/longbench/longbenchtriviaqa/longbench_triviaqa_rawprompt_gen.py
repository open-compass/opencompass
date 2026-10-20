from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    LongBenchF1Evaluator,
    LongBenchtriviaqaDataset,
    triviaqa_postprocess,
)

LongBench_triviaqa_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LongBench_triviaqa_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': 'Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBench_triviaqa_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=triviaqa_postprocess),
)

LongBench_triviaqa_datasets = [
    dict(
        type=LongBenchtriviaqaDataset,
        abbr='LongBench_triviaqa',
        path='opencompass/Longbench',
        name='triviaqa',
        reader_cfg=LongBench_triviaqa_reader_cfg,
        infer_cfg=LongBench_triviaqa_infer_cfg,
        eval_cfg=LongBench_triviaqa_eval_cfg,
    )
]
