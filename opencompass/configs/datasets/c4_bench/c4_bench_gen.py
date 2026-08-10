from opencompass.datasets import C4BenchDataset, C4BenchEvaluator
from opencompass.openicl.icl_inferencer import ChatMLInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever


c4_bench_reader_cfg = dict(
    input_columns=['question'],
    output_column='reference',
)


def _c4_dataset(abbr, split):
    return dict(
        abbr=abbr,
        type=C4BenchDataset,
        path='sci-m-wang/C4-Eval',
        split=split,
        reader_cfg=c4_bench_reader_cfg,
        infer_cfg=dict(
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=ChatMLInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(type=C4BenchEvaluator),
            pred_role='BOT',
        ),
    )


c4_bench_datasets = [_c4_dataset('C4-Bench', 'primary')]

c4_bench_task_datasets = [
    _c4_dataset(f'C4-Bench-{task}', task)
    for task in ('H0', 'H1', 'H4', 'E0', 'E1')
]
