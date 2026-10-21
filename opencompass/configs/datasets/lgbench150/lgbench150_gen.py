"""LG-Bench 150-question graph-structured evaluation subset."""

from pathlib import Path

from opencompass.datasets import CustomDataset, LGBench150Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


DATA_DIR = Path(__file__).resolve().parents[3] / 'datasets' / 'lgbench150_data'

reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer',
)
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=('{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n'
                        'Answer with exactly one letter: A, B, C, or D.'),
            )
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
eval_cfg = dict(
    evaluator=dict(
        type=LGBench150Evaluator,
        questions_path=str(DATA_DIR / 'questions.jsonl'),
        graph_path=str(DATA_DIR / 'graph.jsonl'),
        threshold=0.4,
    ))

lgbench150_datasets = [
    dict(
        type=CustomDataset,
        abbr='lg-bench-150-gen',
        path=str(DATA_DIR),
        file_name='questions.jsonl',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
