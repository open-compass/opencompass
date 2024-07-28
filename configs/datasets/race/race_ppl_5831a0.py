from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import RaceDataset

race_reader_cfg = dict(
    input_columns=['article', 'question', 'A', 'B', 'C', 'D'],
    output_column='answer',
    train_split='validation',
    test_split='test'
)

race_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            ans: dict(
                round=[
                    dict(role='HUMAN', prompt='Article:\n{article}\nQuestion:\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}'),
                    dict(role='BOT', prompt=f'Answer: {ans}'),
                ]
            )
            for ans in ['A', 'B', 'C', 'D']
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

race_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

race_datasets = [
    dict(
        abbr='race-middle',
        type=RaceDataset,
        path='opencompass/race',
        name='middle',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg),
    dict(
        abbr='race-high',
        type=RaceDataset,
        path='opencompass/race',
        name='high',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg)
]
