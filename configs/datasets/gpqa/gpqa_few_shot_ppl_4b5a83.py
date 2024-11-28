from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import GPQADataset, GPQAEvaluator
from opencompass.utils import first_option_postprocess

gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

hint = f'For the multiple choice question below, please provide the correct answer option directly.'
question_and_options = 'Question: {question}\n(A){A}\n(B){B}\n(C){C}\n(D){D}\n'
gpqa_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template={
        opt: f'{question_and_options}\nAnswer: {opt}' for opt in ['A', 'B', 'C', 'D']},
        ),
    prompt_template=dict(
        type=PromptTemplate,
        template={
        opt: f'{hint}\n</E>{question_and_options}\nAnswer: {opt}' for opt in ['A', 'B', 'C', 'D']
        },
        ice_token='</E>'
        ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=PPLInferencer))

gpqa_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator))

gpqa_datasets = []
gpqa_subsets = {
    # 'extended': 'gpqa_extended.csv',
    # 'main': 'gpqa_main.csv',
    'diamond': 'gpqa_diamond.csv'
}

for split in list(gpqa_subsets.keys()):
    gpqa_datasets.append(
        dict(
            abbr='GPQA_' + split,
            type=GPQADataset,
            path='./data/gpqa/',
            name=gpqa_subsets[split],
            reader_cfg=gpqa_reader_cfg,
            infer_cfg=gpqa_infer_cfg,
            eval_cfg=gpqa_eval_cfg)
    )
