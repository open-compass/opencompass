from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import HellaswagDatasetwithICE
from opencompass.utils.text_postprocessors import first_capital_postprocess

hellaswag_reader_cfg = dict(
    input_columns=['ctx', 'A', 'B', 'C', 'D'],
    output_column='label',
    train_split='train',
    test_split='val',
)

hint = 'Continue the following text without adding any additional information or formatting:'
question_and_options = '{ctx}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nWhat is the right option?'
hellaswag_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template={answer: f'{question_and_options}\n{answer}\n' for answer in ['A', 'B', 'C', 'D']},
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template={answer: f'{hint}\n</E>{question_and_options}\n{answer}' for answer in ['A', 'B', 'C', 'D']},
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=list(range(10))),
    inferencer=dict(type=PPLInferencer),
)

hellaswag_eval_cfg = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_postprocessor=dict(type=first_capital_postprocess),
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
