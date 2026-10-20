from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import RaceDataset

race_reader_cfg = dict(
    input_columns=['article', 'question', 'A', 'B', 'C', 'D'],
    output_column='answer',
    train_split='validation',
    test_split='test'
)

hint = 'Read the article, and answer the question by replying A, B, C or D.'
question_and_options = '{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}'
race_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={answer: hint + '\n\n' + question_and_options + '\n\nAnswer: ' + answer for answer in ['A', 'B', 'C', 'D']}),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

race_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator))

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
