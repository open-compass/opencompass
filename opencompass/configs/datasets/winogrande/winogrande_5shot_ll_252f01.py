from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import LLInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import WinograndeDatasetV3

winogrande_reader_cfg = dict(
    input_columns=['opt1', 'opt2'],
    output_column='answer',
    train_split='train_xs',
    test_split='dev',
)

question_and_options = 'Which of the following is a good sentence:\nA. {opt1}\nB. {opt2}'
winogrande_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template={'A': '{opt1}', 'B': '{opt2}'},
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template={'A': '</E>{opt1}', 'B': '</E>{opt2}'},
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 2, 4, 6, 8]),
    inferencer=dict(type=LLInferencer),
)
winogrande_eval_cfg = dict(evaluator=dict(type=AccwithDetailsEvaluator))

winogrande_datasets = [
    dict(
        abbr='winogrande',
        type=WinograndeDatasetV3,
        path='opencompass/winogrande',
        reader_cfg=winogrande_reader_cfg,
        infer_cfg=winogrande_infer_cfg,
        eval_cfg=winogrande_eval_cfg)
]
