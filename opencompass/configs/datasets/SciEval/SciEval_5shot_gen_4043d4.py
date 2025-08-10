from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.datasets import SciEvalDataset  

# 只评测 biology + multiple-choice 的 test split
_hint = ('Given a question and four options, please select the right answer. '
         "Your answer should be 'A', 'B', 'C' or 'D'.")
category = [
    'biology',
]

scieval_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='test',
)

scieval_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
            ),
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

scieval_eval_cfg = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

scieval_datasets = [
    dict(
        abbr='scieval_biology',
        type=SciEvalDataset,
        path='OpenDFM/SciEval',
        name='default',
        category=category, 
        reader_cfg=scieval_reader_cfg,
        infer_cfg=scieval_infer_cfg,
        eval_cfg=scieval_eval_cfg,
    )
]
