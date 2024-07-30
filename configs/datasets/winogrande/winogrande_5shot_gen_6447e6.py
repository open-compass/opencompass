from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WinograndeDatasetV3
from opencompass.utils.text_postprocessors import first_option_postprocess

winogrande_reader_cfg = dict(
    input_columns=['opt1', 'opt2'],
    output_column='answer',
    train_split='train_xs',
    test_split='dev',
)

winogrande_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt='Which of the following is a good sentence:\nA. {opt1}\nB. {opt2}\nAnswer:'),
                dict(role='BOT', prompt='{answer}'),
            ]
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 2, 4, 6, 8]),
    inferencer=dict(type=GenInferencer),
)

winogrande_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

winogrande_datasets = [
    dict(
        abbr='winogrande',
        type=WinograndeDatasetV3,
        path='opencompass/winogrande',
        reader_cfg=winogrande_reader_cfg,
        infer_cfg=winogrande_infer_cfg,
        eval_cfg=winogrande_eval_cfg,
    )
]
