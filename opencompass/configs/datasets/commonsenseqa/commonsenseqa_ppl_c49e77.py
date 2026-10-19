# Use FixKRetriever to avoid hang caused by the Huggingface
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import commonsenseqaDataset

commonsenseqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D', 'E'],
    output_column='answerKey',
    test_split='validation')

_ice_template = dict(
    type=PromptTemplate,
    template={
        ans: dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt='Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nAnswer: '),
                dict(role='BOT', prompt=f'{ans}'),
            ])
        for ans in ['A', 'B', 'C', 'D', 'E']
    },
    ice_token='</E>')

commonsenseqa_infer_cfg = dict(
    ice_template=_ice_template,
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4, 5, 6, 7]),
    inferencer=dict(type=PPLInferencer))

commonsenseqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

commonsenseqa_datasets = [
    dict(
        abbr='commonsense_qa',
        type=commonsenseqaDataset,
        path='opencompass/commonsense_qa',
        reader_cfg=commonsenseqa_reader_cfg,
        infer_cfg=commonsenseqa_infer_cfg,
        eval_cfg=commonsenseqa_eval_cfg)
]
