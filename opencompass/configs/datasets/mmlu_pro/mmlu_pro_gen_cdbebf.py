from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUProDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

with read_base():
    from .mmlu_pro_categories import categories


mmlu_pro_datasets = []

for category in categories:
    mmlu_pro_reader_cfg = dict(
        input_columns=['question', 'cot_content', 'options_str'],
        output_column='answer',
        train_split='validation',
        test_split='test',
    )

    mmlu_pro_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt='Question:\n{question}\nOptions:\n{options_str}'),
                dict(role='BOT', prompt="Answer: Let's think step by step. {cot_content}")
            ]),
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(role='HUMAN', prompt='Question:\n{question}\nOptions:\n{options_str}'),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer),
    )

    mmlu_pro_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCDEFGHIJKLMNOP'),
    )

    mmlu_pro_datasets.append(
        dict(
            abbr=f'mmlu_pro_{category.replace(" ", "_")}',
            type=MMLUProDataset,
            category=category,
            reader_cfg=mmlu_pro_reader_cfg,
            infer_cfg=mmlu_pro_infer_cfg,
            eval_cfg=mmlu_pro_eval_cfg,
        ))
