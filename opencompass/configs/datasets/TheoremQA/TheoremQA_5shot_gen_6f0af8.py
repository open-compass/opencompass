from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TheoremQADatasetV3, TheoremQA_postprocess_v3, TheoremQAEvaluatorV3

with read_base():
    from .TheoremQA_few_shot_examples import examples

num_shot = 5
rounds = []
for index, (query, response) in enumerate(examples[:num_shot]):
    if index == 0:
        desc = 'You are supposed to provide a solution to a given problem.\n\n'
    else:
        desc = ''
    rounds += [
        dict(role='HUMAN', prompt=f'{desc}Problem:\n{query}\nSolution:'),
        dict(role='BOT', prompt=f'{response}')
    ]
rounds += [dict(role='HUMAN', prompt='Problem:\n{Question}\nSolution:')]

TheoremQA_reader_cfg = dict(input_columns=['Question', 'Answer_type'], output_column='Answer', train_split='test', test_split='test')

TheoremQA_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=dict(round=rounds)),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024, stopping_criteria=['Problem:', 'Problem']),
)

TheoremQA_eval_cfg = dict(
    evaluator=dict(type=TheoremQAEvaluatorV3),
    pred_postprocessor=dict(type=TheoremQA_postprocess_v3)
)

TheoremQA_datasets = [
    dict(
        abbr='TheoremQA',
        type=TheoremQADatasetV3,
        path='data/TheoremQA/theoremqa_test.json',
        reader_cfg=TheoremQA_reader_cfg,
        infer_cfg=TheoremQA_infer_cfg,
        eval_cfg=TheoremQA_eval_cfg,
    )
]
