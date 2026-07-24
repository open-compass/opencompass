from opencompass.datasets import FakeAlignmentDataset
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

fake_alignment_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='category',
)

fake_alignment_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(role='system', content='You are a helpful assistant.'),
            dict(role='user', content='{prompt}'),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

fake_alignment_eval_cfg = dict(
    evaluator=dict(
        type='FakeAlignmentJudgeEvaluator',
    ),
)

fake_alignment_datasets = [
    dict(
        abbr='fake_safety',
        type=FakeAlignmentDataset,
        path='opencompass/fake_alignment/safety.jsonl',
        reader_cfg=fake_alignment_reader_cfg,
        infer_cfg=fake_alignment_infer_cfg,
        eval_cfg=fake_alignment_eval_cfg,
    ),
    dict(
        abbr='dna_training_set',
        type=FakeAlignmentDataset,
        path='opencompass/fake_alignment/dna_training_set.jsonl',
        reader_cfg=fake_alignment_reader_cfg,
        infer_cfg=fake_alignment_infer_cfg,
        eval_cfg=fake_alignment_eval_cfg,
    ),
]
