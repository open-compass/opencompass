from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MMLUProDataset, MMLUProBaseEvaluator

with read_base():
    from .mmlu_pro_categories import categories

mmlu_pro_datasets = []

for category in categories:
    hint = f'Answer the following multiple choice question about {category}, and give your answer option directly.'
    question_and_options = 'Question:\n{question}\nOptions:\n{options_str}'
    mmlu_pro_reader_cfg = dict(
        input_columns=['question', 'cot_content', 'options_str'],
        output_column='answer_string',
        train_split='validation',
        test_split='test',
    )
    mmlu_pro_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=f'{question_and_options}\nAnswer: {{answer}}'),
        prompt_template=dict(
            type=PromptTemplate,
            template=f'{hint}\n</E>{question_and_options}\nAnswer: ',
            ice_token='</E>'
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
            inferencer=dict(type=GenInferencer, max_out_len=100)
    )

    mmlu_pro_eval_cfg = dict(
        evaluator=dict(type=MMLUProBaseEvaluator)
    )

    mmlu_pro_datasets.append(
        dict(
            abbr=f'mmlu_pro_{category.replace(" ", "_")}',
            type=MMLUProDataset,
            path='opencompass/mmlu_pro',
            category=category,
            reader_cfg=mmlu_pro_reader_cfg,
            infer_cfg=mmlu_pro_infer_cfg,
            eval_cfg=mmlu_pro_eval_cfg,
        ))
