from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import HellaswagDatasetwithICE
from opencompass.utils.text_postprocessors import first_option_postprocess

hellaswag_reader_cfg = dict(
    input_columns=['ctx', 'A', 'B', 'C', 'D'],
    output_column='label',
    train_split='train',
    test_split='val',
)

hellaswag_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=f'{{ctx}}\nA) {{A}}\nB) {{B}}\nC) {{C}}\nD) {{D}}\nWhat is the right option?'),
                dict(role='BOT', prompt='{label}\n'),
            ]
        ),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='HUMAN', prompt='Continue the following text without adding any additional information or formatting:\n'),
                '</E>',
            ],
            round=[
                dict(role='HUMAN', prompt=f'{{ctx}}\nA) {{A}}\nB) {{B}}\nC) {{C}}\nD) {{D}}\nWhat is the right option?'),
                dict(role='BOT', prompt='{label}\n'),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=list(range(10))),
    inferencer=dict(type=GenInferencer),
)

hellaswag_eval_cfg = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

hellaswag_datasets = [
    dict(
        abbr='hellaswag',
        type=HellaswagDatasetwithICE,
        path='opencompass/hellaswag_ice',
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg,
    )
]
