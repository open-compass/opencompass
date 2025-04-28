from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import MMLUDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

# 指定只使用生命科学相关的子集
mmlu_life_science_subsets = [
    'anatomy',                # 解剖学
    'clinical_knowledge',     # 临床知识
    'professional_medicine',  # 专业医学
    'medical_genetics',       # 遗传学
    'college_medicine',       # 大学医学
    'college_biology',        # 大学生物学
]

mmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev')

mmlu_datasets = []

for _name in mmlu_life_science_subsets:
    _hint = f'There is a single choice question about {_name.replace("_", " ")}. Answer the question by replying A, B, C or D.'
    mmlu_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '),
                dict(role='BOT', prompt='{target}\n')
            ]),
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
                    ),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer),
    )

    mmlu_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

    mmlu_datasets.append(
        dict(
            abbr=f'lukaemon_mmlu_{_name}',
            type=MMLUDataset,
            path='opencompass/mmlu',
            name=_name,
            reader_cfg=mmlu_reader_cfg,
            infer_cfg=mmlu_infer_cfg,
            eval_cfg=mmlu_eval_cfg,
        ))

del _name, _hint
